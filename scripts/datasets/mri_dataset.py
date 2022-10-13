import os
import logging
from random import choices
from collections import Counter
from abc import ABC, abstractclassmethod
from typing import Optional, List, Dict, Tuple, TypedDict

import pandas as pd
from tqdm import tqdm

import nibabel as nib

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import traceback


class MRIPatientData(TypedDict):
    image: Optional[torch.Tensor]
    label: Optional[torch.Tensor]
    path: Optional[Dict[str, str]]


class MRIPatientBaseDataset(Dataset, ABC):
    """Patch Base Dataset. Abstract class.

    Patch datasets treats each patch to be independent

    Attributes:
        data_root_: str containing the root path of the dataset
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
                 df: pd.DataFrame,
                 mri_modalities: List[str],
                 transform: callable,
                 target_transform: callable = torch.tensor,
                 primary_label_func: callable = lambda x: x[0],
                 balance_patch_per_class: bool = False) -> None:
        """Inits MRI Patient Classification Dataset"""

        self.data_root_ = data_root
        self.df_ = df
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.primary_label_func_ = primary_label_func

        self.all_mri_modalities_ = ["flair", "t1", "t1ce", "t2"]
        self.selected_mri_modalities_ = mri_modalities
        assert not set(self.selected_mri_modalities_).difference(
            self.all_mri_modalities_)

        self.get_im_path = lambda p, m: os.path.join(self.data_root_, p,
                                                     f"{p}_{m}.nii.gz")
        insts = [
            self.check_one_patient(r)
            for _, r in tqdm(self.df_.iterrows(), total=len(self.df_))
        ]
        self.instances_ = [i for i in insts if i[0]]

        if balance_patch_per_class:
            self.replicate_balance_instances(self.primary_label_func_)

        self.get_weights(self.primary_label_func_)

    def process_classes(self, tgt_func: Optional[callable] = None):
        """Look for all the labels in the dataset

        Creates the classes_, and class_to_idx_ attributes"""
        if not tgt_func: tgt_func = self.primary_label_func_
        all_labels = [tgt_func(i[1]) for i in self.instances_]

        self.classes_ = sorted(set(all_labels))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: {}".format(self.classes_))
        if len(self.classes_) <= 1:
            logging.warning(
                "Less than 2 classes exist. Check data if training")

    def get_weights(self, tgt_func: Optional[callable] = None):
        """Count number of instances for each class, and computes weights"""
        # Get classes
        if not tgt_func: tgt_func = self.primary_label_func_
        self.process_classes(tgt_func)
        all_lbl = [self.class_to_idx_[tgt_func(i[1])] for i in self.instances_]

        # Count number of slides in each class
        count = Counter(all_lbl)
        count = torch.Tensor([count[i] for i in range(len(count))])
        logging.info("Count: {}".format(count))

        # Compute weights
        inv_count = 1 / count
        self.weights_ = inv_count / torch.sum(inv_count)
        logging.debug("Weights: {}".format(self.weights_))
        return self.weights_

    def replicate_balance_instances(self, primary_label_func):
        """resample the instances list to balance each class"""
        if not primary_label_func:
            primary_label_func = self.primary_label_func_
        all_labels = [primary_label_func(si[1]) for si in self.instances_]
        val_sample = max(Counter(all_labels).values())

        all_instances_ = []
        for l in sorted(set(all_labels)):
            instances_l = [
                si for si in self.instances_ if primary_label_func(si[1]) == l
            ]
            all_instances_.extend(choices(instances_l, k=val_sample))

        self.instances_ = all_instances_

    @abstractclassmethod
    def check_one_patient(self, row: pd.Series) -> Tuple:
        ...

    @abstractclassmethod
    def __getitem__(self, idx: int) -> MRIPatientData:
        ...

    def __len__(self) -> int:
        return len(self.instances_)

    def get_mri_instance_fnames(self, row: pd.Series):
        pt_id = row["patient"]
        out = {"patient": pt_id}
        out.update({
            m: self.get_im_path(pt_id, m)
            for m in self.selected_mri_modalities_
        })
        for m in self.selected_mri_modalities_:
            if not os.path.exists(out[m]):
                logging.ERROR(f"{pt_id} series {m} does not exist.")
                raise RuntimeError(f"{pt_id} series {m} does not exist.")
        return out, self.get_gt(row)


class MRIPatientVolumeClassificationDataset(MRIPatientBaseDataset):
    def __init__(self,
                 data_root: str,
                 df: pd.DataFrame,
                 mri_modalities: List[str],
                 transform: callable,
                 target_transform: callable = torch.tensor,
                 primary_label_func: callable = lambda x: x[0],
                 balance_patch_per_class: bool = False) -> None:
        """Inits MRI Patient Classification Dataset"""

        # using [4:] to exclude institution, patient, mosaic, patch_code
        self.get_gt = lambda row: row[4:].to_list()

        super().__init__(data_root=data_root,
                         df=df,
                         mri_modalities=mri_modalities,
                         transform=transform,
                         target_transform=target_transform,
                         primary_label_func=primary_label_func,
                         balance_patch_per_class=balance_patch_per_class)

    def check_one_patient(self, row: pd.Series) -> Tuple[Dict[str, str], List]:
        return self.get_mri_instance_fnames(row)

    def __getitem__(self, idx: int) -> MRIPatientData:
        imps, target = self.instances_[idx]
        target = self.class_to_idx_[self.primary_label_func_(target)]
        get_im = lambda fn: torch.tensor(nib.load(fn).get_fdata()) / 32767.
        try:
            im = torch.cat(
                [get_im(imps[m]) for m in self.selected_mri_modalities_],
                dim=2).swapaxes(-1, 0).to(torch.float32)
        except:
            logging.critical("bad patient - {}".format(imps["patient"]))
            print(traceback.format_exc())
            return MRIPatientData()

        if self.transform_ is not None:
            im = self.transform_(im)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return MRIPatientData(image=im, label=target, path=imps)

    def __len__(self) -> int:
        return len(self.instances_)


class MRIPatientAxialSliceClassificationDataset(MRIPatientBaseDataset):
    def __init__(self,
                 data_root: str,
                 df: pd.DataFrame,
                 mri_modalities: List[str],
                 transform: callable,
                 target_transform: callable = torch.tensor,
                 primary_label_func: callable = lambda x: x[0],
                 balance_patch_per_class: bool = False) -> None:
        """Inits MRI Patient Classification Dataset"""

        # using [4:] to exclude institution, patient, mosaic, patch_code
        self.get_gt = lambda row: row[4:].to_list()

        super().__init__(data_root=data_root,
                         df=df,
                         mri_modalities=mri_modalities,
                         transform=transform,
                         target_transform=target_transform,
                         primary_label_func=primary_label_func,
                         balance_patch_per_class=balance_patch_per_class)

    def check_one_patient(self, row: pd.Series) -> Tuple[Dict[str, str], List]:
        paths, gt = self.get_mri_instance_fnames(row)
        seg_mask = nib.load(self.get_im_path(row["patient"],
                                             "seg")).get_fdata()
        max_area_slice_idx = (seg_mask > 0).sum(axis=(0, 1)).argmax()
        return paths, gt, max_area_slice_idx

    def __getitem__(self, idx: int) -> MRIPatientData:
        imps, target, slice_idx = self.instances_[idx]
        target = self.class_to_idx_[self.primary_label_func_(target)]
        get_im = lambda fn: torch.tensor(nib.load(fn).get_fdata(
        ))[:, :, slice_idx].unsqueeze(0) / 32767.
        try:
            im = torch.cat(
                [get_im(imps[m]) for m in self.selected_mri_modalities_],
                dim=0).to(torch.float32)
        except:
            logging.critical("bad patient - {}".format(imps["patient"]))
            print(traceback.format_exc())
            return MRIPatientData()

        if self.transform_ is not None:
            im = self.transform_(im)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return MRIPatientData(image=im, label=target, path=imps)

    def __len__(self) -> int:
        return len(self.instances_)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])

    logging.info("MRIPatientClassificationDataset Debug Log")

    mpcd = MRIPatientAxialSliceClassificationDataset(
        data_root="/Volumes/umms-tocho/data/brats/brats2021",
        df=pd.read_csv(
            "/Volumes/umms-tocho/code/chengjia/torchsrh/torchsrh/train/data/brats/brats21_train.csv"
        ),
        mri_modalities=["t1ce"],
        transform=transforms.Compose([transforms.RandomHorizontalFlip()]))

    all_im = [d['image'] for d in tqdm(mpcd)]
    import pdb
    pdb.set_trace()

    from torchsrh.models.resnet_embedding import resnet_emb
    model = resnet_emb(num_channel_in=3, pretrained=False, arch="resnet50")
    model.forward(data["image"].unsqueeze(0))
