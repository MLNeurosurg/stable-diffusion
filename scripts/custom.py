import os
import pandas as pd
import numpy as np
import albumentations
import torch
from torch.utils.data import Dataset
import time
from tifffile import imread
from torchvision import transforms
from torchvision.transforms import (Compose, ToTensor, Normalize, RandomAffine,
                                    RandomApply, RandomHorizontalFlip,
                                    RandomVerticalFlip, ColorJitter,
                                    GaussianBlur, RandomErasing, RandomCrop)

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
class GetThirdChannel(torch.nn.Module):
    """Computes the third channel of SRH image
    Compute the third channel of SRH images by subtracting CH3 and CH2. The
    channel difference is added to the subtracted_base.
    """
    def __init__(self, subtracted_base: float = 5000 / 65536.0):
        self.subtracted_base = subtracted_base

    def __call__(self, two_channel_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            two_channel_image: a 2 channel np array in the shape H * W * 2
            subtracted_base: an integer to be added to (CH3 - CH2)
        Returns:
            A 3 channel np array in the shape H * W * 3
        """
        ch2 = two_channel_image[0, :, :]
        ch3 = two_channel_image[1, :, :]
        ch1 = ch3 - ch2 + self.subtracted_base

        return torch.stack((ch1, ch2, ch3), dim=0)

class MinMaxChop(torch.nn.Module):
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_ = min_val
        self.max_ = max_val

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.clamp(self.min_, self.max_)

class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            RandomCrop(256),
            transforms.Normalize((0,0),(2**16,2**16)),
            GetThirdChannel(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            MinMaxChop()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path_ = self.data[index]
        path_ = "/nfs/turbo/umms-tocho/root_srh_db"+ path_
        try:
            img = imread(path_)
        except FileNotFoundError:
            time.sleep(10)
            img = imread(path_)
        img = np.float32(img)
        img = np.moveaxis(img, 0, -1)
        img = self.transform(img)
        # img = torch.moveaxis(img, 0, 1)
        # print(img.shape)

        return img



class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file):
        super().__init__()
        df = pd.read_csv(training_images_list_file)
        self.data = list(df["file_name"])


class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file):
        super().__init__()
        df = pd.read_csv(test_images_list_file)
        self.data = list(df["file_name"])


