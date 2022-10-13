import os

from glob import glob
import pandas as pd
from typing import Optional, List, Dict, Tuple, TypedDict
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
from collections import Counter
from random import choices
import copy


class MultimodalEmbeddingData(TypedDict):
    mri_emb: torch.Tensor
    srh_emb: torch.Tensor
    label: torch.Tensor
    mri_emb_path: Dict[str, str]
    srh_emb_path: Dict[str, str]
    mri_id: str
    srh_id: str


class MultimodalEmbeddingDataset(Dataset):

    def __init__(self,
                 srh_data_root: str,
                 mri_data_root: str,
                 df: pd.DataFrame,
                 balance_patch_per_class: bool = False):
        self.srh_data_root_ = srh_data_root
        self.mri_data_root_ = mri_data_root
        self.df_ = df

        self.balance_patch_per_class_ = balance_patch_per_class

        self.df_.drop(
            ["mosaic", "brats_mosaic", "patch_code", "brats_patch_code"],
            axis="columns",
            inplace=True)
        self.df_.drop_duplicates(inplace=True)
        make_mri_path = lambda x: os.path.join(self.mri_data_root_, f"{x}*.pt")
        make_srh_path = lambda x: os.path.join(self.srh_data_root_, f"{x}*.pt")

        def proc_one_row(row):
            return ({
                "mri_emb_path": glob(make_mri_path(row['brats_patient'])),
                "srh_emb_path": glob(make_srh_path(row['patient'])),
                "mri_id": row['brats_patient'],
                "srh_id": row['patient']
            }, row["idh"])

        self.instances_ = [
            proc_one_row(r)
            for _, r in tqdm(self.df_.iterrows(), total=len(self.df_))
        ]

        if self.balance_patch_per_class_:
            self.replicate_balance_instances(lambda x: x)
        self.process_classes()
        self.get_weights()

    def __getitem__(self, idx: int) -> MultimodalEmbeddingData:

        data, target = self.instances_[idx]
        data = copy.deepcopy(data)
        data["label"] = torch.tensor(int(target))

        data["mri_emb"] = torch.cat(
            [torch.load(f) for f in data["mri_emb_path"]]).squeeze()
        data["srh_emb"] = torch.cat(
            [torch.load(f) for f in data["srh_emb_path"]]).mean(dim=0)
        data["srh_emb_path"] = "#".join(data["srh_emb_path"])
        return data

    def process_classes(self, tgt_func: Optional[callable] = lambda x: x):
        """Look for all the labels in the dataset

        Creates the classes_, and class_to_idx_ attributes"""
        all_labels = [tgt_func(i[1]) for i in self.instances_]
        self.classes_ = sorted(set(all_labels))
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
        logging.info("Labels: {}".format(self.classes_))

    def get_weights(self, tgt_func: Optional[callable] = lambda x: x):
        """Count number of instances for each class, and computes weights"""
        # Get classes
        self.process_classes(tgt_func)
        all_labels = [
            self.class_to_idx_[tgt_func(i[1])] for i in self.instances_
        ]

        # Count number of slides in each class
        count = Counter(all_labels)
        count = torch.Tensor([count[i] for i in range(len(count))])
        logging.info("Count: {}".format(count))

        # Compute weights
        inv_count = 1 / count
        self.weights_ = inv_count / torch.sum(inv_count)
        logging.debug("Weights: {}".format(self.weights_))
        return self.weights_

    def replicate_balance_instances(
            self, primary_label_func: Optional[callable] = lambda x: x):
        """resample the instances list to balance each class"""

        all_labels = [primary_label_func(si[1]) for si in self.instances_]
        val_sample = max(Counter(all_labels).values())

        all_instances_ = []
        for l in sorted(set(all_labels)):
            instances_l = [
                si for si in self.instances_ if primary_label_func(si[1]) == l
            ]
            all_instances_.extend(choices(instances_l, k=val_sample))

        self.instances_ = all_instances_

    def __len__(self):
        return len(self.instances_)


if __name__ == "__main__":
    med = MultimodalEmbeddingDataset(
        mri_data_root=
        "/Volumes/umms-tocho/exp/chengjia/mm_mri_ce_embs/9b9d6c5e_mri/embeddings",
        srh_data_root=
        "/Volumes/umms-tocho/exp/chengjia/mm_he_ce_embs/276f1699_tcga/embeddings",
        df=pd.read_csv(
            "/Volumes/umms-tocho/code/chengjia/torchsrh/torchsrh/train/data/tcga_multimodal/mm_tcga_brats_val.csv"
        ))
    data_0 = med.__getitem__(0)
    import pdb
    pdb.set_trace()
