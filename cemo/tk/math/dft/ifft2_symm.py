import torch
from typing import Tuple, List, Union
from cemo.tk.math.dft.ifftn_symm import ifftn_symm
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def ifft2_symm(x: Tensor, dims: TuLi = (-2, -1)) -> Tensor:
    """
    2D inverse Fourier transform with a symmetric coordinate layout.

    Args:
        x: input tensor
        dims: dimensions to be transformed

    Returns:
        Real space data with a symmetric coordinate layout
    """
    return ifftn_symm(x, dims=dims)
