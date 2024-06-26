import torch
import math
from torch import Tensor
from typing import Optional


def add_gaussian_noise(
        img: Tensor,
        snr: float,
        debug: bool = False,
        mask: Optional[Tensor] = None):
    """
    Add Gaussian noise to a 2D image(s).

    Args:
        img: input image(s) of shape (L, L) or (N, L, L)
            where N is the total number of images.
        snr: target signal-to-noise ratio (SNR)
        debug: print debugging msgs (default False)
        mask: a binary mask of shape (L, L) or (N, L, L)
    Return:
        images with noise added according the desired SNR
    """
    if mask is not None:
        mask_ready = mask.expand(img.shape)
        img_ready = img[mask_ready]
        if debug:
            print(">>>> Apply mask to input images")
            print(">>> mask[mask] length ", mask[mask].shape)
    else:
        img_ready = img
    if debug:
        print("img_ready.shape", img_ready.shape)
    img_std = torch.std(img_ready, unbiased=True).item()
    noise_std = img_std / math.sqrt(snr)
    noise = torch.normal(0, noise_std, size=img.shape).to(img.device)
    if debug:
        print(f"target snr: {snr}")
        print(f"input image std: {img_std}")
        print(f"noise std (target): {noise_std}")
        print(f"generated noise mean: {noise.mean()}")
        print(f"generated noise std: {noise_std}")
        print("----------------------------------")
    return img + noise
