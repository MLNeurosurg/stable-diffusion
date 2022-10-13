import os
import logging
from collections import namedtuple, Counter
from typing import Optional, List, Union, Dict, Any, Tuple, TypedDict, NamedTuple
import random

import itertools
import math

import random
import logging
import numpy as np
from tifffile import imread

get_chnl_min = lambda im: im.min(dim=1).values.min(dim=1).values.squeeze()
get_chnl_max = lambda im: im.max(dim=1).values.max(dim=1).values.squeeze()


def patch_code_to_list(patch_code: int) -> List[str]:
    """Converts patch code to a list of patch types

    Patch codes are an integer from 0-7 (3 bits), and patch types can be
    "nondiagnostic", "normal", or "tumor". When the conversion is performed,
    we first convert the patch code into bits that correspond to
    ["nondiagnostic", "normal", "tumor"] in this order. The effect can be
    summerized using this chart:
    | patch code | patch type list                      |
    | ---------- | ------------------------------------ |
    | 0          | []                                   |
    | 1          | ['tumor']                            |
    | 2          | ['normal']                           |
    | 3          | ['normal', 'tumor']                  |
    | 4          | ['nondiagnostic']                    |
    | 5          | ['nondiagnostic', 'tumor']           |
    | 6          | ['nondiagnostic', 'normal']          |
    | 7          | ['nondiagnostic', 'normal', 'tumor'] |
    | 99         | ['all']                              |
    Patch code == 99 is a special case, used for studies where the 3 class
    classification is not avaliable.

    Args:
        patch_code: an integer in the rane [0,7] U {99}

    Returns:
        A list of strings representing the patch types that ccorrespond to the
        patch code. See conversion chart above.
    """
    if patch_code == 99:
        return ["all"]

    patch_code = [int(i) for i in bin(patch_code)[2:].zfill(3)]
    patch_name = ["nondiagnostic", "normal", "tumor"]
    return [n for (c, n) in zip(patch_code, patch_name) if c]


def ceil_instance_thres_archive(instances: List[Any], thres: int):
    """random sample thres number of instances from instances list

    If thres >= len(instances), we randomly sample from instances.
    If thres < len(instances), we shuffle, repeat instances, and then choose
    the first thres items.

    Args:
        instances: a list of instances
        thres: the threshold for number of instances(slides/patches) to be
            sampled / oversampled
        """
    raise NotImplementedError()
    num_repeat = math.ceil(thres / len(instances))
    random.shuffle(instances)
    instances_repeated = list(
        itertools.chain(*itertools.repeat(instances, num_repeat)))
    return sorted(instances_repeated[:thres])


class PatchInstance(NamedTuple):
    """A patch internal representation

    Attributes:
        label: a list of labels
        im_path: path to image
        patch_name: the filename of the image (tiff filename)
    """
    label: str
    im_path: str
    patch_name: str

    def __len__(self) -> int:
        return 1


class SlideInstance(NamedTuple):
    """A mosaic, containing its labels and patches
    Attributes:
        name: the name of the slide
        label: a list of labels, can by of any type
        patches: a list of paths to patches in the slide
    """

    name: str
    label: str
    patches: List[PatchInstance]

    def __len__(self) -> int:
        return len(self.patches)
