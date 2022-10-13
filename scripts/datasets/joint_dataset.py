import os
import logging
from collections import namedtuple, Counter
from typing import Tuple, Optional, Any, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset


class JointDataset(Dataset):
    def __init__(self, src: Dataset, tgt: Dataset) -> None:
        self.src_dst_ = src
        self.tgt_dst_ = tgt

        assert hasattr(self.src_dst_, 'classes_')
        assert hasattr(self.tgt_dst_, 'classes_')
        assert self.src_dst_.classes_ == self.tgt_dst_.classes_
        self.classes_ = self.src_dst_.classes_

        assert hasattr(self.src_dst_, 'weights_')
        assert hasattr(self.tgt_dst_, 'weights_')
        self.src_weights_ = self.src_dst_.weights_
        self.tgt_weights_ = self.tgt_dst_.weights_
        assert self.src_weights_.shape == self.tgt_weights_.shape

    def __len__(self) -> int:
        return max(len(self.src_dst_), len(self.tgt_dst_))

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        src_data = self.src_dst_.__getitem__(idx % len(self.src_dst_))
        tgt_data = self.tgt_dst_.__getitem__(idx % len(self.tgt_dst_))

        return {
            "src_data": src_data.data,
            "src_label": src_data.label,
            "src_file": "",
            "tgt_data": tgt_data.data,
            "tgt_label": tgt_data.label,
            "tgt_file": ""
        }


if __name__ == "__main__":
    from torchsrh.datasets import PatchDataset

    csv_root = "/nfs/turbo/umms-tocho/code/chengjia/torchsrh-contrastive/train/data/"
    nio_csv_path = os.path.join(csv_root, "srh5v0.1/db_nio_train_1000.csv")
    inv_csv_path = os.path.join(csv_root, "srh5v0.1/db_inv_train_1000.csv")
    data_root = "/nfs/turbo/umms-tocho/root_srh_db/"
    inv_dset = PatchDataset(data_root=data_root, slides_file=inv_csv_path)
    nio_dset = PatchDataset(data_root=data_root, slides_file=nio_csv_path)
    jd = JointDataset(inv_dset, nio_dset)

    jd_data = jd.__getitem__(10)
    import pdb
    pdb.set_trace()
