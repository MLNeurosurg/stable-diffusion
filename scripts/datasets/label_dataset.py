import os
import logging
from collections import Counter
from typing import NamedTuple, Optional, List
from itertools import chain

import pandas
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import is_image_file
from torchvision import transforms

from torchsrh.datasets.patch_dataset import PatchData
from torchsrh.datasets.db_improc import process_read_srh
from torchsrh.datasets.common import (patch_code_to_list, ceil_instance_thres,
                                      get_chnl_min, get_chnl_max)

from torchsrh.datasets.slide_dataset import Slide
from torchsrh.datasets.slide_dataset import walk_all_slides, get_ids_patches_for_minibatch


class Label(NamedTuple):
    """A label"""

    label: List
    patches: List[str]

    def __len__(self) -> int:
        return len(self.patches)

    def __repr__(self) -> str:
        return "Label(label={}, len={})".format(self.label, len(self))


class LabelBaseDataset(Dataset):
    """Label Base Dataset. Abstract class.

    Attributes:
        data_root_: str containing the root path of the dataset
        image_reader_: callable that reads in images (and some processing)
        transform_: transformations to be performed on the data
        target_transform_: transformations to be performed on the labels
        df_: data frame containing patch information
        instances_: a list of Slide
        classes_: a set of primary labels in the dataset (could be any
            hashable object)
        class_to_idx_: a mapping from the primary class label to a numeric
            label [0 .. num classes - 1]
        weights_: weights assigned to each class, inverse proportional to the
            slide count in each class
    """
    def __init__(self,
                 data_root: str,
                 df: pandas.DataFrame,
                 image_reader: callable = process_read_srh,
                 transform: Optional[callable] = transforms.ToTensor(),
                 target_transform: Optional[callable] = torch.tensor) -> None:
        """Inits the base abstract dataset

        Populate each attribute and walk through each label to look for patches
        """
        self.data_root_ = data_root
        self.image_reader_ = image_reader
        self.transform_ = transform
        self.target_transform_ = target_transform

        self.df_ = df
        self.instances_ = []

        self.generate_instances()

        assert len(self.instances_)

    def process_classes(self, tgt_func: callable = lambda x: x[0]):
        """Look for all the labels in the dataset

        Creates the classes_, and class_to_idx_ attributes"""
        all_labels = [tgt_func(i.label) for i in self.instances_]
        self.classes_ = sorted(set(all_labels))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.debug("Labels: {}".format(self.classes_))

    def get_weights(self, tgt_func: callable = lambda x: x[0]):
        """Count number of instances for each class, and computes weightes"""
        # Get classes
        self.process_classes(tgt_func)
        all_labels = [
            self.class_to_idx_[tgt_func(i.label)] for i in self.instances_
        ]

        # Count number of labels in each class
        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(count))])
        logging.debug("Count: {}".format(count))

        # Compute weights
        inv_count = 1 / count
        self.weights_ = inv_count / torch.sum(inv_count)
        logging.debug("Weights: {}".format(self.weights_))

        return self.weights_

    def generate_instances(self,
                           tgt_func: callable = lambda x: x[0],
                           patch_code: Optional[int] = 1):
        """Pull patches for each label and validate the images"""
        lname_to_slides = {}

        # iterate through each slide
        for i in range(len(self.df_)):

            df = self.df_.loc[i]
            gt = df.loc[df.index[4]:df.index[-1]].values.tolist()

            lname_to_slides[tgt_func(gt)] = []

        # fill lname_to_slides
        lname_to_slides = walk_all_slides(self.df_, lname_to_slides,
                                          self.data_root_, tgt_func, "label",
                                          patch_code)

        # create label instances
        for label in lname_to_slides.keys():

            if lname_to_slides[label]:
                curr_slides = lname_to_slides[label]
                curr_label = curr_slides[0].label
                assert all(x.label == curr_label for x in curr_slides)
                label_patches = list(chain(*(x.patches for x in curr_slides)))

                self.instances_.append(Label(curr_label, label_patches))
                logging.debug("Label {} OK".format(label))
            else:
                logging.error("Label {} all empty".format(label))
                pass

    def __len__(self):
        return len(self.instances_)


class LabelDataset(LabelBaseDataset):
    """Label Dataset. Treats each label to be independent.

    Same attributes as LabelBaseDataset, with an additional one listed below.

    Attributes:
        primary_label_func_: callable to select the the primary label from a list
    """
    def __init__(self,
                 data_root: str,
                 slides_file: str,
                 image_reader: callable = process_read_srh,
                 transform: Optional[callable] = transforms.ToTensor(),
                 target_transform: Optional[callable] = torch.tensor,
                 primary_label_func: callable = lambda x: x[0]) -> None:
        """Inits the label dataset

        Populate each attribute and walk through each label to look for patches.
        The constructor takes in a path to a CSV slide file
        """
        df = pandas.read_csv(slides_file)
        super().__init__(data_root, df, image_reader, transform,
                         target_transform)
        self.primary_label_func_ = primary_label_func
        self.get_weights(tgt_func=primary_label_func)

    def __getitem__(self, idx: int):
        """Retrieve a patch from label specified by idx"""
        label = self.instances_[idx]
        target = self.class_to_idx_[self.primary_label_func_(label.label)]

        imp = np.random.permutation(label.patches)[0]

        try:
            im = self.image_reader_(imp)
        except:
            logging.error("bad_file - {}".format(imp))
            return None, None

        if self.transform_ is not None:
            im = self.transform_(im)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return PatchData(im, target)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S')
