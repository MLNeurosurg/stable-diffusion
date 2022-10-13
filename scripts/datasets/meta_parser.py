import os
import json
import math
import random
import logging
import itertools
from typing import List, Any

import pandas
from tqdm import tqdm

from torch.utils.data import Dataset

from scripts.datasets.common import patch_code_to_list
from scripts.datasets.common import PatchInstance, SlideInstance
from enum import Enum, unique, auto


@unique
class DatasetLevel(Enum):
    PATCH = auto()
    PATCH_EMBEDDING = auto()
    SLIDE = auto()
    PATIENT = auto()


class SRHCSVParser():

    def __init__(self, dset: Dataset, df: pandas.DataFrame):
        self.dset_ = dset
        self.df_ = df

        slide_list = df.apply(
            lambda x: ".".join([x["patient"], str(x["mosaic"])]),
            axis=1).tolist()
        assert len(slide_list) == len(set(slide_list))  # slides are unique

    @staticmethod
    def make_gt_list(x) -> List[Any]:
        return x.iloc[4:].tolist()

    def get_gt_col_name(self, df: pandas.DataFrame) -> str:
        return self.dset_.primary_label_func_(
            self.make_gt_list(pandas.Series(df.keys())))

    def get_gt(self, x: pandas.Series, patch_code: str) -> str:  # x is a row
        assert patch_code in {None, "tumor", "normal", "nondiagnostic"}

        if patch_code in {None, "tumor"}:
            return self.dset_.primary_label_func_(self.make_gt_list(x))
        else:
            return patch_code

    def get_patch_instances(self):
        if self.dset_.dataset_level_ is None:
            self.dset_.dataset_level_ = DatasetLevel.PATCH

        for (inst_name, patient_id), patient_s in tqdm(
                self.df_.groupby(["institution", "patient"])):

            self.dset_.instances_ += SRHMetaParser(
                inst_name=inst_name,
                patient_id=patient_id,
                csv_parser=self,
                dataset_level=self.dset_.dataset_level_).process_slides(
                    patient_s)

    def get_slide_instances(self):
        if self.dset_.dataset_level_ is None:
            self.dset_.dataset_level_ = DatasetLevel.SLIDE

        for (inst_name, patient_id), patient_s in tqdm(
                self.df_.groupby(["institution", "patient"])):

            mp = SRHMetaParser(inst_name=inst_name,
                               patient_id=patient_id,
                               csv_parser=self,
                               dataset_level=self.dset_.dataset_level_)
            for _, slide_s in patient_s.iterrows():
                slide_instance = SlideInstance(
                    name=f"{patient_id}/{slide_s['mosaic']}",
                    label=self.get_gt(slide_s, None),
                    patches=mp.process_slide(slide_s))
                if len(slide_instance):
                    self.dset_.instances_.append(slide_instance)

    def get_patient_instances(self):
        if self.dset_.dataset_level_ is None:
            self.dset_.dataset_level_ = DatasetLevel.PATIENT

        group_keys = ["institution", "patient", self.get_gt_col_name(self.df_)]
        for (inst_name, patient_id,
             prime_label), patient_s in tqdm(self.df_.groupby(group_keys)):
            mp = SRHMetaParser(inst_name=inst_name,
                               patient_id=patient_id,
                               csv_parser=self,
                               dataset_level=self.dset_.dataset_level_)

            patient_instance = SlideInstance(
                name=patient_id,
                label=prime_label,
                patches=mp.process_slides(patient_s))
            if len(patient_instance):
                self.dset_.instances_.append(patient_instance)


class SRHMetaParser():
    """Parser to work with MLiNS internal SRH dataset metadata files

    It works with json metadata files for each patient and requires dataframes
    to describe the slides **in a single patient** that we want. The dataframe
    should have the format:
    ```
    institution, patient, mosaic, patch_code, label1, label2, ...
    ```
    It produces list of instances, which are dictionaries in the format
    specified by make_instance_dict function.

    Attributes:
        data_root_: str containing the root path of the dataset
        seg_model_: str specifying the hash of the segmentation model
        p_meta_: Dict read in from the json metadata file
    """

    def __init__(self,
                 inst_name: str,
                 patient_id: str,
                 csv_parser: SRHCSVParser,
                 dataset_level: DatasetLevel = DatasetLevel.PATCH):
        """Inits the SRH Metadata parser"""
        self.data_root_ = csv_parser.dset_.data_root_
        self.seg_model_ = csv_parser.dset_.seg_model_
        self.slide_patch_thres_ = csv_parser.dset_.slide_patch_thres_
        self.label_parser_ = csv_parser

        meta_file = os.path.join(self.data_root_, inst_name, patient_id,
                                 f"{patient_id}_meta.json")
        with open(meta_file) as fd:
            self.p_meta_ = json.load(fd)

        if dataset_level == DatasetLevel.PATCH_EMBEDDING:
            self.patch_path_func_ = self.make_emb_path
        else:
            self.patch_path_func_ = self.make_im_path

    def process_slides(self, patient_s: pandas.Series):
        return list(
            itertools.chain(*[
                self.process_slide(slide_s)
                for _, slide_s in patient_s.iterrows()
            ]))

    def process_slide(self, slide_s: pandas.Series):
        slide_name = str(slide_s.mosaic)
        if slide_name not in self.p_meta_["slides"]:
            logging.warning(
                f"Slide {slide_s['patient']}/{slide_s['mosaic']} DNE")
            return []

        all_patches_slide = self.p_meta_["slides"][slide_name]["predictions"][
            self.seg_model_]

        patch_code_decoded = patch_code_to_list(slide_s["patch_code"])
        patch_code_diff = set(patch_code_decoded).difference(
            all_patches_slide.keys())
        if patch_code_diff:
            logging.warning(f"Slide {slide_s.patient} - {slide_s.mosaic} " +
                            f"does not have any patches in {patch_code_diff}")

        slide_instances = [
            PatchInstance(im_path=self.patch_path_func_(p_name, slide_s),
                          label=self.label_parser_.get_gt(slide_s, p_code),
                          patch_name=p_name)
            for p_code in patch_code_to_list(slide_s["patch_code"])
            for p_name in all_patches_slide[p_code]
        ]

        if len(slide_instances) > 0:
            logging.debug(
                f"Slide {slide_s['patient']}/{slide_s['mosaic']} OK:" +
                f"{len(slide_instances)} " +
                f"{self.label_parser_.make_gt_list(slide_s)}")
        else:
            logging.debug(
                f"Slide {slide_s['patient']}/{slide_s['mosaic']} Empty")

        if self.slide_patch_thres_:
            slide_instances = self.ceil_instance_thres(slide_instances)
        return slide_instances

    def ceil_instance_thres(self, instances: List[Any]):
        """random sample thres number of instances from instances list

        If thres >= len(instances), we randomly sample from instances.
        If thres < len(instances), we shuffle, repeat instances, and then choose
        the first thres items.

        Args:
            instances: a list of instances
            thres: the threshold for number of instances(slides/patches) to be
                sampled / oversampled
            """

        num_repeat = math.ceil(self.slide_patch_thres_ / len(instances))
        random.shuffle(instances)
        instances_repeated = list(
            itertools.chain(*itertools.repeat(instances, num_repeat)))
        return sorted(instances_repeated[:self.slide_patch_thres_])

    def make_im_path(self, patch_id: str, slide_s: pandas.Series):
        """Parser for patch data.

        Produces path to the image
        """
        path = os.path.join(self.data_root_,
                            slide_s["institution"], slide_s["patient"],
                            str(slide_s.mosaic), "patches", patch_id)
        if not (path.endswith(".tif") or path.endswith(".tiff")):
            path += ".tif"

        return path

    def make_emb_path(self, patch_id: str, slide_s: pandas.Series):
        """Parser for patch data.

        Produces  path to the pt file that stores the data for the patient
        """
        return os.path.join(self.label_parser_.dset_.embed_root_,
                            f"{slide_s['patient']}.{slide_s['mosaic']}.pt")
