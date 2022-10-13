import logging
from typing import Optional, List

import torch
import pandas
import numpy as np
from torchvision import transforms

from torchsrh.datasets.db_improc import process_read_srh
from torchsrh.datasets.balanced_dataset import BalancedDataset
from torchsrh.datasets.common import PatchInstance
from torchsrh.datasets.patch_dataset import SRHCSVParser


class PatientBaseDataset(BalancedDataset):
    """Patient Base Dataset. Abstract class.

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
                 transform: Optional[callable] = transforms.ToTensor(),
                 target_transform: Optional[callable] = torch.tensor,
                 segmentation_model: Optional[str] = "03207B00",
                 slide_patch_thres: Optional[int] = None,
                 balance_patients_class: bool = False,
                 primary_label_func: callable = lambda x: x[0],
                 process_read_im: callable = process_read_srh) -> None:
        """Inits the base abstract dataset

        Populate each attribute and walk through each patient to look for patches
        """
        super().__init__(primary_label_func=primary_label_func)
        self.data_root_ = data_root
        self.seg_model_ = segmentation_model
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.process_read_im_ = process_read_im
        self.slide_patch_thres_ = slide_patch_thres
        self.dataset_level_ = "patient"
        self.instances_ = []
        self.classes_ = None
        self.class_to_idx_ = None
        self.weights_ = None

        SRHCSVParser(self, df).get_patient_instances()
        assert len(self.instances_) > 0
        if balance_patients_class: self.replicate_balance_instances()
        self.get_weights()

    def __len__(self):
        return len(self.instances_)


class PatientDataset(PatientBaseDataset):
    """Patient Dataset. Treats each patient to be independent.

    Same attributes as PatientBaseDataset, with an additional one listed below.

    Attributes:
        primary_label_func_: callable to select the the primary label from a list
    """
    def __init__(self, slides_file: str, **args) -> None:
        """Inits the patient dataset

        Populate each attribute and walk through each patient to look for patches.
        The constructor takes in a path to a CSV slide file
        """

        super().__init__(df=pandas.read_csv(slides_file), **args)

    def __getitem__(self, idx: int):
        """Retrieve a patch from patient specified by idx"""
        slide = self.instances_[idx]
        target = self.class_to_idx_[slide.label]

        # randomly choose a patch from the wholeslide
        patch = slide.patches[np.random.choice(np.arange(len(slide)))]

        # Read image
        try:
            im = self.process_read_im_(patch.im_path)
        except:
            logging.error(f"bad_file - {patch['im_path']}")
            return {}

        # Perform transformations
        if self.transform_ is not None:
            im = self.transform_(im)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "path": [patch.im_path]}


class PatientContrastiveDataset(PatientBaseDataset):
    """Patient Dataset. Treats each patient to be independent.

    Same attributes as PatientBaseDataset, with an additional one listed below.

    Attributes:
        primary_label_func_: callable to select the the primary label from a list
    """
    def __init__(self,
                 slides_file: str,
                 num_samples: int = 2,
                 num_transforms: int = 2,
                 **args) -> None:
        """Inits the patient dataset

        Populate each attribute and walk through each patient to look for patches.
        The constructor takes in a path to a CSV slide file
        """

        super().__init__(df=pandas.read_csv(slides_file), **args)
        self.num_samples_ = num_samples
        self.num_transforms_ = num_transforms

    def read_images(self, inst: List[PatchInstance]):
        """Read in a list of patches, different patches and transformations"""
        im_id = np.random.permutation(np.arange(len(inst)))
        images = []
        imps_take = []

        idx = 0
        while len(images) < self.num_samples_:
            curr_inst = inst[im_id[idx % len(im_id)]]
            try:
                images.append(self.process_read_im_(curr_inst.im_path))
                imps_take.append(curr_inst.im_path)
                idx += 1
            except:
                logging.error("bad_file - {}".format(curr_inst.im_path))

        assert self.transform_ is not None
        xformed_im = [
            self.transform_(im) for _ in range(self.num_transforms_)
            for im in images
        ]
        return xformed_im, imps_take

    def __getitem__(self, idx: int):
        """Retrieve a list of patches, from the wholeslide specified by idx"""
        slide = self.instances_[idx]
        target = self.class_to_idx_[slide.label]
        im, imp = self.read_images(slide.patches)

        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "path": [imp]}


if __name__ == '__main__':
    from torchsrh.datasets.db_improc import get_transformations
    logging.basicConfig(
        level=logging.DEBUG,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])
    logging.info("Patch Data Debug Log")

    csv_path = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh/torchsrh/train/data/srh7v1/srh7v1_train.csv"
    data_root = "/nfs/turbo/umms-tocho/root_srh_db/"
    tx, vx = get_transformations()
    dset = PatientContrastiveDataset(data_root=data_root,
                                     slides_file=csv_path,
                                     segmentation_model="03207B00",
                                     transform=tx,
                                     balance_patients_class=True)
    out = dset.__getitem__(10)
    import pdb; pdb.set_trace() #yapf:disable