
import torch
from typing import Optional, Iterable
from cemo.tk.math import dft
from typing import Tuple, List, Union
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def fourier_filter(
        x: Tensor,
        dims: TuLi,
        domain: str,
        mask: Optional[Tensor] = None,
        use_rfft: bool = False,
        s: Optional[Iterable[int]] = None,
        symm_freq: bool = False,
        symm_real: bool = True,
        ) -> Tensor:
    """
    Fourier-space filtering of images.

    Args:
        x: input images in real space (type: Tensor)
        dims: dimensions to be transformed
        domain: domain of input x: "fourier" or "real"
        mask: mask (type: Tensor or None)
        use_rfft: whether to use rfft or fft
        s: full sizes of the Fourier-transform dimensions
        symm_freq: whether to use a symmetric layout in Fourier space
        symm_real: whether to use a symmetric layout in real space

    Returns:
        a tuple: (images, images_ft)
        images: Fourier-space filtered images (type: Tensor)
        images_ft: Fourier transform of images (type: Tensor)
    """
    if symm_freq and use_rfft:
        raise ValueError("symm and use_rfft cannot both be True")

    if domain == "real":
        images_ft = dft.fftn(
            x,
            dims=dims,
            use_rfft=use_rfft,
            symm_freq=symm_freq,
            symm_real=symm_real)
    elif domain == "fourier":
        images_ft = x
    else:
        raise ValueError(f"Invalidate domain: {domain}")

    if mask is None:
        new_images_ft = images_ft
    else:
        new_images_ft = images_ft * mask.to(dtype=images_ft.dtype)

    new_images_real = dft.ifftn(
        new_images_ft,
        dims=dims,
        use_rfft=use_rfft,
        symm_freq=symm_freq,
        symm_real=symm_real,
        s=s,
        ).real

    return (new_images_real, new_images_ft)
