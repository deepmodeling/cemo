import torch
from typing import Iterable
from cemo.tk.transform import transform_images
from cemo.tk.math import dht, dft
Tensor = torch.Tensor


def imgshift(
        image: Tensor,
        shift_ratio: Tensor,
        s: Iterable[int],
        domain: str,
        dims: Iterable[int] = (-2, -1),
        is_rfft: bool = False,
        indexing: str = "ij",
        reverse: bool = False,
        symm: bool = False,
        debug: bool = False) -> Tensor:
    """
    Shift images in Fourier, Hartley, or real space.

    Args:
        image: input images (B, H, W)
        s: shape of the full-size image (2,)
        shift_ratio: shift ratio (B, 2)
        domain: Fourier, Hartley, or real space
        is_rfft: whether to use rfft or fft
        indexing: order of frequency mesh grid coordinates
            (default "ij", i.e., (x, y); the alternative is "xy", i.e., (y, x).
        reverse: whether to reverse the shift direction
        symm: whether to use a symmetric freq-grid layout
        debug: whether to print debug information

    Returns:
        shifted image(s) of shape (B, H, W)
    """
    device = image.device
    full_image_sizes = torch.tensor(
        [x for x in s],
        ).reshape(1, 2).to(device=device)  # shape: (1, 2)
    shift_ratio = shift_ratio.view(-1, 2)

    if domain == "real":
        shift = shift_ratio
    else:
        shift_px = shift_ratio * full_image_sizes  # (B, 2)
        shift = shift_px

    if reverse:
        shift = -1. * shift

    if domain == "fourier":
        output = dft.fourier_shift(
            x=image,
            shift=shift,
            dims=dims,
            s=s,
            indexing=indexing,
            is_rfft=is_rfft,
            symm=symm,
            debug=debug)

    elif domain == "hartley":
        # shape of output: (B, H, W)
        output = dht.hartley_shift(
            image,
            shift=shift,
            dims=dims,
            indexing=indexing,
            symm=symm,
            )

    elif domain == "real":
        image = image.float()
        output = transform_images(
            image,
            shift_ratio=shift,
            interp_mode="bilinear",
            align_corners=True)

    else:
        raise ValueError("domain must be one of 'fourier', 'hartley', 'real'")
    return output
