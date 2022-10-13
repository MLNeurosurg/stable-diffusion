import math
import random
import logging

from collections import Counter

import torch
from torch.utils.data import Dataset
from datasets.meta_parser import DatasetLevel


class BalancedDataset(Dataset):
    """Balanced Base Dataset.

    Datasets that allows data from each class to be balanced

    Attributes:
        classes_: a set of primary labels in the dataset (could be any
            hashable object)
        class_to_idx_: a mapping from the primary class label to a numeric
            label [0 .. num classes - 1]
        weights_: weights assigned to each class, inverse proportional to the
            slide count in each class
    """
    def __init__(self, primary_label_func: callable = lambda x: x[0]):
        super().__init__()

        self.primary_label_func_ = primary_label_func

        self.dataset_level_ = None
        self.classes_ = None
        self.class_to_idx_ = None
        self.weights_ = None

    def process_classes(self):
        """Look for all the labels in the dataset

        Creates the classes_, and class_to_idx_ attributes"""
        all_labels = [i.label for i in self.instances_]
        self.classes_ = sorted(set(all_labels))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: {}".format(self.classes_))

    def get_weights(self):
        """Count number of instances for each class, and computes weights"""
        # Get classes
        self.process_classes()
        all_labels = [self.class_to_idx_[i.label] for i in self.instances_]

        # Count number of slides in each class
        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(count))])
        logging.info("Count: {}".format(count))

        # Compute weights
        inv_count = 1.0 / count
        self.weights_ = inv_count / torch.sum(inv_count)
        logging.debug("Weights: {}".format(self.weights_))

        return self.weights_

    def replicate_balance_instances(self):
        """resample the instances list to balance each class"""
        random.shuffle(self.instances_)
        all_labels = [i.label for i in self.instances_]
        val_sample = max(Counter(all_labels).values())

        all_repl_instances = []
        for l in sorted(set(all_labels)):
            inst_l = [i for i in self.instances_ if i.label == l]
            n_rep = math.ceil(val_sample / len(inst_l))
            all_repl_instances.extend((inst_l * n_rep)[:val_sample])

        self.instances_ = all_repl_instances
        if self.dataset_level_ in {
                DatasetLevel.PATCH, DatasetLevel.PATCH_EMBEDDING
        }:
            self.instances_.sort(key=lambda x: x.im_path)
        else:
            self.instances_.sort(key=lambda x: x.name)