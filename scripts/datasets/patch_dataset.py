import math
import random
import logging
from abc import ABC
from typing import Optional, List, TypedDict, Any, NamedTuple

import torch
import pandas
import numpy as np
from tqdm import tqdm

from datasets.db_improc import process_read_srh
from datasets.meta_parser import SRHCSVParser, DatasetLevel
from datasets.balanced_dataset import BalancedDataset
from datasets.common import get_chnl_min, get_chnl_max
from tifffile import imread

class PatchData(TypedDict):
    image: Optional[torch.Tensor]
    label: Optional[torch.Tensor]
    path: Optional[List[str]]


class PatchContrastiveData(TypedDict):
    image: Optional[List[torch.Tensor]]
    label: Optional[torch.Tensor]
    path: Optional[List[str]]


class PatchBaseDataset(BalancedDataset, ABC):
    """Patch Base Dataset.

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
                 df: pandas.DataFrame,
                 transform: callable,
                 target_transform: callable = torch.tensor,
                 segmentation_model: Optional[str] = "03207B00",
                 slide_patch_thres: Optional[int] = None,
                 balance_patch_per_class: bool = False,
                 primary_label_func: callable = lambda x: x[0],
                 process_read_im: callable = process_read_srh,
                 dataset_level: DatasetLevel = DatasetLevel.PATCH) -> None:
        """Inits the base abstract dataset

        Populate each attribute and walk through each slide to look for patches
        """
        super().__init__(primary_label_func=primary_label_func)
        self.data_root_ = data_root
        self.seg_model_ = segmentation_model
        self.transform_ = transform
        self.target_transform_ = target_transform
        self.slide_patch_thres_ = slide_patch_thres
        self.process_read_im_ = process_read_im
        self.dataset_level_ = dataset_level

        self.instances_ = []
        self.classes_ = None
        self.class_to_idx_ = None
        self.weights_ = None

        SRHCSVParser(self, df).get_patch_instances()
        if len(self.instances_) == 0:
            logging.warning("dataset empty")
        if balance_patch_per_class: self.replicate_balance_instances()
        self.get_weights()

    def __len__(self):
        return len(self.instances_)

    def __getitem__(self, index):
        raise NotImplementedError()


class PatchDFDataset(PatchBaseDataset):
    """Patch DF Dataset.

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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def __getitem__(self, idx: int) -> PatchData:
        """Retrieve a patch specified by idx"""
        inst = self.instances_[idx]
        target = self.class_to_idx_[inst.label]

        try:
            im = imread(inst.im_path)
        except:
            logging.error("bad_file - {}".format(inst.im_path))
            return {"image": None, "label": None, "path": [None]}

        # logging.debug(f"before xform im shape {im.shape}")
        # logging.debug(f"before xform im mean  {im.mean(dim=[1,2])}")
        # logging.debug(f"before xform im min   {get_chnl_min(im)}")
        # logging.debug(f"before xform im max   {get_chnl_max(im)}")
        im = np.float32(im)
        im = np.moveaxis(im, 0, -1)
        # Perform transformations
        if self.transform_ is not None:
            im = self.transform_(im)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        if isinstance(im, torch.Tensor):
            logging.debug(f"after xform im shape {im.shape}")
            logging.debug(f"after xform im mean  {im.mean(dim=[1,2])}")
            logging.debug(f"after xform im min   {get_chnl_min(im)}")
            logging.debug(f"after xform im max   {get_chnl_max(im)}")

        return {"image": im, "label": target, "path": [inst.im_path]}


class PatchDataset(PatchDFDataset):
    """Patch Dataset. Treats each patch to be independent.

    Same attributes as PatchBaseDataset, with an additional one listed below.
    """

    def __init__(self, slides_file: str, **kwargs) -> None:
        """Inits the patch dataset

        Populate each attribute and walk through each slide to look for patches.
        The constructor takes in a path to a CSV slide file
        """

        super().__init__(df=pandas.read_csv(slides_file), **kwargs)


class PatchEmbeddingDataset(PatchDFDataset):
    """Patch Dataset. Treats each patch to be independent.

    Same attributes as PatchBaseDataset, with an additional one listed below.
    """

    def __init__(self, embedding_root: str, slides_file: str,
                 **kwargs) -> None:
        """Inits the patch dataset

        Populate each attribute and walk through each slide to look for patches.
        The constructor takes in a path to a CSV slide file
        """
        self.embed_root_ = embedding_root
        super().__init__(df=pandas.read_csv(slides_file),
                         dataset_level=DatasetLevel.PATCH_EMBEDDING,
                         **kwargs)

        self.all_mapping_ = {}
        for p in set([i.im_path for i in self.instances_]):
            try:
                data = torch.load(p)
            except:
                logging.error(f"embedding file {p} DNE")
                continue

            self.all_mapping_[p.split("/")[-1]] = {
                id.split("/")[-1]: i
                for (i, id) in enumerate(data["path"])
            }

    def __getitem__(self, idx: int) -> PatchContrastiveData:
        inst = self.instances_[idx]
        target = self.class_to_idx_[inst.label]

        try:
            emb = torch.load(inst.im_path)["embeddings"][self.all_mapping_[
                inst.im_path.split("/")[-1]][inst.patch_name]]
        except:
            logging.error("bad_file - {}".format(inst.im_path))
            return {"image": None, "label": None, "path": [None]}

        if self.transform_ is not None:
            emb = self.transform_(emb)
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {
            "image": emb,
            "label": target,
            "path": [inst.im_path],
            "patch_id": [inst.patch_name]
        }


class PatchContrastiveDataset(PatchBaseDataset):
    """Patch Contrastive Dataset. Performs multiple transformations on a patch

    Can be used for SimCLR and SupCon experiments

    Same attributes as SlideBaseDataset, with an additional two listed below
    """

    def __init__(self, slides_file: str, num_transforms: int = 2, **kwargs) -> None:
        """Inits the patch contrastive dataset"""
        super().__init__(df=pandas.read_csv(slides_file), **kwargs)
        self.num_transforms_ = num_transforms

    def __getitem__(self, idx: int) -> PatchContrastiveData:
        inst = self.instances_[idx]
        target = self.class_to_idx_[inst.label]

        try:
            im = self.process_read_im_(inst.im_path)
        except:
            logging.error("bad_file - {}".format(inst.im_path))
            return {"image": None, "label": None, "path": [None]}

        if self.transform_ is not None:
            im = [self.transform_(im) for _ in range(self.num_transforms_)]
        if self.target_transform_ is not None:
            target = self.target_transform_(target)

        return {"image": im, "label": target, "path": [inst.im_path]}


if __name__ == '__main__':
    from datasets.db_improc import get_transformations
    logging.basicConfig(
        level=logging.DEBUG,
        format=
        "[%(levelname)-s|%(asctime)s|%(filename)s:%(lineno)d|%(funcName)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler()])
    logging.info("Patch Data Debug Log")

    csv_path = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh/torchsrh/train/data/srh7v1/srh7v1_train.csv"
    data_root = "/nfs/turbo/umms-tocho/root_srh_db/"

    csv_path = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh/torchsrh/train/data/srh7v1/srh7v1_test.csv"
    embed_root = "/nfs/turbo/umms-tocho/exp/chengjia/sampling/simclr_adamw_finetune/61490973-Jul10-12-54-35-36dd585b-1-4-0.001_1000_iter_iter/evals/5673ca36-Sep05-23-58-11-dev_/predictions/"
    tx, vx = get_transformations()
    dset = PatchEmbeddingDataset(data_root=data_root,
                                 embedding_root=embed_root,
                                 slides_file=csv_path,
                                 segmentation_model="03207B00",
                                 transform=None,
                                 balance_patch_per_class=True)
    out = dset.__getitem__(10)
    import pdb; pdb.set_trace() #yapf:disable
