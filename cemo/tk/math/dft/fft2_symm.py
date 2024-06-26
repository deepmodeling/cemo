import torch
from typing import Tuple, List, Union
from cemo.tk.math.dft.fftn_symm import fftn_symm
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def fft2_symm(x: Tensor, dims: TuLi = (-2, -1)) -> Tensor:
    """
    2D Fourier transform with a symmetric frequency coordinate layout.

    Args:
        x: input tensor
        dims: dimensions to be transformed

    Returns:
        Fourier coefficients with a symmetric frequency coordinate layout
    """
    return fftn_symm(x, dims=dims)
