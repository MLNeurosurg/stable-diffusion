#import pytest
import torch
from torchsrh.datasets.db_improc import FFTLowPassFilter, MinMaxChop, LaserNoise


def test_fft_low_pass_mask_5_11():
    fftlpf = FFTLowPassFilter(circ_radius=[5, 5])
    mask_out = fftlpf.circ_mask(torch.Size((11, 11)))
    mask_exp = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert (mask_out.to(int) - mask_exp).sum() < 1e-6


def test_fft_low_pass_mask_5_rect():
    fftlpf = FFTLowPassFilter(circ_radius=[5, 5])
    mask_out = fftlpf.circ_mask(torch.Size((15, 11)))
    mask_exp = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert (mask_out.to(int) - mask_exp).sum() < 1e-6


def test_fft_low_pass_mask_5_12():
    fftlpf = FFTLowPassFilter(circ_radius=[5, 5])
    mask_out = fftlpf.circ_mask(torch.Size((12, 12)))
    print(mask_out.to(int))
    mask_exp = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                             [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                             [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert (mask_out.to(int) - mask_exp).sum() < 1e-6


def test_fft_low_pass_identity():
    fftlpf = FFTLowPassFilter(circ_radius=[17, 17])
    mask_out = fftlpf.circ_mask(torch.Size((12, 12)))
    mask_exp = torch.ones(torch.Size((12, 12)))
    assert (mask_out.to(int) - mask_exp).sum() < 1e-6

    for _ in range(10):
        rand = torch.rand(torch.Size((300, 300)))
        assert (fftlpf.__call__(rand) - rand).sum() < 1e-2


def test_min_max_chop():
    chop = MinMaxChop()
    im = chop(torch.Tensor([-1, 0, 0.2, 0.5, 0.8, 1, 2]))
    assert (im - torch.Tensor([0, 0, 0.2, 0.5, 0.8, 1, 1])).sum() < 1e-6


def test_laser_noise_runs():
    ln = LaserNoise(shot_noise_min_rate=0.0,
                    shot_noise_max_rate=0.0,
                    scatter_min_var=0.0,
                    scatter_max_var=1.0)
    im = torch.zeros((300, 300))
    aug_im = ln(im)
    assert aug_im.sum() < 1e-6
