import random
import logging
from functools import partial
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
from tifffile import imread
from skimage.filters import gaussian

import torch
from torch.nn import ModuleList
from torch.fft import fft2, fftshift, ifft2, ifftshift
import torchvision
from torchvision.transforms import (
    Compose, Resize, Normalize, RandomAffine, RandomApply,
    RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, GaussianBlur,
    RandomErasing, RandomAutocontrast, RandomEqualize, RandomSolarize,
    RandomPosterize, RandomAdjustSharpness, Grayscale, RandomResizedCrop)


class GetThirdChannel(torch.nn.Module):
    """Computes the third channel of SRH image

    Compute the third channel of SRH images by subtracting CH3 and CH2. The
    channel difference is added to the subtracted_base.

    """

    def __init__(self, subtracted_base: float = 5000 / 65536.0):
        super().__init__()
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
        super().__init__()
        self.min_ = min_val
        self.max_ = max_val

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.clamp(self.min_, self.max_)


class GaussianNoise(torch.nn.Module):
    """Object to add guassian noise to images."""

    def __init__(self, min_var: float = 0.01, max_var: float = 0.1):
        super().__init__()
        self.min_var = min_var
        self.max_var = max_var

    def __call__(self, tensor):

        var = random.uniform(self.min_var, self.max_var)
        noisy = tensor + torch.randn(tensor.size()) * var
        noisy = torch.clamp(noisy, min=0., max=1.)
        return noisy


class LaserNoise(torch.nn.Module):
    """Object to add laser noise to images."""

    def __init__(self,
                 shot_noise_min_rate: float = 0.0,
                 shot_noise_max_rate: float = 0.2,
                 scatter_min_var: float = 0.0,
                 scatter_max_var: float = 1.0):
        super().__init__()
        self.shot_noise_min_rate = shot_noise_min_rate
        self.shot_noise_max_rate = shot_noise_max_rate
        self.scatter_min_var = scatter_min_var
        self.scatter_max_var = scatter_max_var

    def __call__(self, img: torch.Tensor):  # 2 channels
        # sample a sigma value for gaussian blur of noise
        sigma_val = random.uniform(2, 3)

        # additive shot noise
        var_shot = random.uniform(self.shot_noise_min_rate,
                                  self.shot_noise_max_rate)
        shot_noise = torch.randn(img.size()) * var_shot
        shot_noise = gaussian(shot_noise, sigma=sigma_val, channel_axis=None)

        # multiplicative scatter noise
        var_mul = random.uniform(self.scatter_min_var, self.scatter_max_var)
        scatter_noise = torch.randn(img.size()) * var_mul
        # scatter_noise = gaussian(scatter_noise, sigma=sigma_val, multichannel=False)

        # apply noise to image
        noisy = (img + shot_noise) + (img * scatter_noise)
        # noisy = torch.clamp(noisy, min=0., max=1.)

        return noisy


class FFTLowPassFilter(torch.nn.Module):

    def __init__(self, circ_radius: List[int] = [50, 300]):
        super().__init__()
        self.circ_radius_ = circ_radius

    def circ_mask(self, im_size: torch.Size()):
        im_size = np.array(im_size)
        r = torch.linspace(0, im_size[-2] - 1, steps=im_size[-2])
        c = torch.linspace(0, im_size[-1] - 1, steps=im_size[-1])
        center = im_size[-2:] // 2
        R, C = torch.meshgrid(r, c, indexing="ij")
        R_diff = R - center[0]
        C_diff = C - center[1]
        radius = random.randint(self.circ_radius_[0], self.circ_radius_[1])
        return (R_diff * R_diff + C_diff * C_diff) < (radius * radius)

    def __call__(self, img: torch.Tensor):  # 3 channels
        fft_img = fftshift(fft2(img))
        fft_img = fft_img * self.circ_mask(img.shape)

        # invert the fft to get the reconstructed images
        freq_filt_img = ifft2(ifftshift(fft_img))
        freq_filt_img = torch.abs(freq_filt_img)
        #freq_filt_img /= freq_filt_img.max()
        return freq_filt_img


class FFTHighPassFilter(torch.nn.Module):

    def __init__(self, circ_radius: List[int] = [50, 300]):
        super().__init__()
        self.circ_radius_ = circ_radius

    def circ_mask(self, im_size: torch.Size()):
        im_size = np.array(im_size)
        r = torch.linspace(0, im_size[-2] - 1, steps=im_size[-2])
        c = torch.linspace(0, im_size[-1] - 1, steps=im_size[-1])
        center = im_size[-2:] // 2
        R, C = torch.meshgrid(r, c, indexing="ij")
        R_diff = R - center[0]
        C_diff = C - center[1]
        radius = random.randint(self.circ_radius_[0], self.circ_radius_[1])
        return (R_diff * R_diff + C_diff * C_diff) >= (radius * radius)

    def __call__(self, img: torch.Tensor):  # 3 channels
        fft_img = fftshift(fft2(img))
        fft_img = fft_img * self.circ_mask(img.shape)

        # invert the fft to get the reconstructed images
        freq_filt_img = ifft2(ifftshift(fft_img))
        freq_filt_img = torch.abs(freq_filt_img)
        #freq_filt_img /= freq_filt_img.max()
        return freq_filt_img


class FFTBandPassFilter(torch.nn.Module):

    def __init__(self,
                 lower_radius: List[int] = [50, 100],
                 higher_radius: List[int] = [150, 300]):
        super().__init__()
        self.lower_radius_ = lower_radius
        self.higher_radius_ = higher_radius

    def circ_mask(self, im_size: torch.Size()):
        im_size = np.array(im_size)
        r = torch.linspace(0, im_size[-2] - 1, steps=im_size[-2])
        c = torch.linspace(0, im_size[-1] - 1, steps=im_size[-1])

        center = im_size[-2:] // 2
        R, C = torch.meshgrid(r, c, indexing="ij")
        R_diff = R - center[0]
        C_diff = C - center[1]
        radius_mat = (R_diff * R_diff + C_diff * C_diff)

        low_radius = random.randint(self.lower_radius_[0],
                                    self.lower_radius_[1])
        high_radius = random.randint(self.higher_radius_[0],
                                     self.higher_radius_[1])

        hpf_mask = radius_mat >= (low_radius * low_radius)
        lpf_mask = radius_mat < (high_radius * high_radius)
        return lpf_mask & hpf_mask

    def __call__(self, img: torch.Tensor):  # 3 channels
        fft_img = fftshift(fft2(img))
        fft_img = fft_img * self.circ_mask(img.shape)

        # invert the fft to get the reconstructed images
        freq_filt_img = ifft2(ifftshift(fft_img))
        freq_filt_img = torch.abs(freq_filt_img)
        #freq_filt_img /= freq_filt_img.max()
        return freq_filt_img


def get_srh_base_aug(aug_dict, rand_prob) -> List:
    u16_min = (0, 0)
    u16_max = (65536, 65536)  # 2^16

    xform_list = [Normalize(u16_min, u16_max)]

    if "laser_noise" in aug_dict:
        ln = LaserNoise(**aug_dict["laser_noise"]["params"])
        xform_list.append(RandomApply(ModuleList([ln]), rand_prob))

    xform_list += [GetThirdChannel(), MinMaxChop()]
    return xform_list


def get_he_base_aug(_, __) -> List:
    return []


def get_strong_aug(augs, rand_prob) -> List:
    """Strong augmentations for OpenSRH training"""
    rand_apply_p = lambda which, **kwargs: RandomApply(
        ModuleList([which(**kwargs)]), p=rand_prob)

    callable_dict = {
        "resize": Resize,
        "random_horiz_flip": partial(RandomHorizontalFlip, p=rand_prob),
        "random_vert_flip": partial(RandomVerticalFlip, p=rand_prob),
        "gaussian_noise": partial(rand_apply_p, which=GaussianNoise),
        "color_jitter": partial(rand_apply_p, which=ColorJitter),
        "random_autocontrast": partial(RandomAutocontrast, p=rand_prob),
        "random_solarize": partial(RandomSolarize, p=rand_prob),
        "random_sharpness": partial(RandomAdjustSharpness, p=rand_prob),
        "drop_color": partial(rand_apply_p, which=Grayscale),
        "gaussian_blur": partial(rand_apply_p, GaussianBlur),
        "random_erasing": partial(RandomErasing, p=rand_prob),
        "random_affine": partial(rand_apply_p, RandomAffine),
        "random_resized_crop": partial(rand_apply_p, RandomResizedCrop),
        "fft_low_pass_filter": partial(rand_apply_p, FFTLowPassFilter),
        "fft_high_pass_filter": partial(rand_apply_p, FFTHighPassFilter),
        "fft_band_pass_filter": partial(rand_apply_p, FFTBandPassFilter),
        #"posterize": partial(RandomPosterize, p=rand_prob), TODO
        #"equalize": partial(RandomEqualize, p=rand_prob) TODO
    }

    return [callable_dict[a["which"]](**a["params"]) for a in augs]


def get_transformations(
    cf: Optional[Dict] = None,
    modality_base_aug_func: Optional[callable] = get_srh_base_aug
) -> Tuple[Compose, Compose]:

    if cf:
        train_dict = cf["data"]["train_augmentation"]
        valid_dict = cf["data"]["valid_augmentation"]
        aug_prob = cf["data"]["augmentation_random_prob"]
    else:
        train_dict = {}
        valid_dict = {}
        aug_prob = 0

    if valid_dict == "same":
        valid_dict = train_dict

    compose_augs = lambda aug_dict, rand_prob: Compose(
        modality_base_aug_func(aug_dict, rand_prob) + get_strong_aug(
            aug_dict, rand_prob))

    train_xform = compose_augs(train_dict, aug_prob)
    valid_xform = compose_augs(valid_dict, aug_prob)

    logging.info(f"train_xform\n{train_xform}")
    logging.info(f"valid_xform\n{valid_xform}")

    return train_xform, valid_xform


def process_read_srh(imp: str) -> torch.Tensor:
    """Read in two channel image

    Args:
        imp: a string that is the path to the tiff image

    Returns:
        A 2 channel torch Tensor in the shape 2 * H * W
    """
    # reference: https://github.com/pytorch/vision/blob/49468279d9070a5631b6e0198ee562c00ecedb10/torchvision/transforms/functional.py#L133

    return torch.from_numpy(imread(imp).astype(np.float32)).contiguous()


def process_read_png(imp: str) -> torch.Tensor:
    """Read in two channel image and get the third channel.

    Args:
        imp: a string that is the path to the tiff image

    Returns:
        A 3 channel np array in the shape 3 * H * W, scaled in range [0, 1]
    """
    # https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html
    return torchvision.transforms.functional.to_tensor(Image.open(imp))
