import torch
from typing import Tuple, List, Union
from cemo.tk.math.dft.fftn_asymm import fftn_asymm
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def fftn_symm(x: Tensor, dims: TuLi, use_rfft: bool) -> Tensor:
    """
    N-dimensional symmetric fast Fourier transform.

    Args:
        x: input tensor
        dims: dimensions to be transformed
        use_rfft: whether to use rfft

    Returns:
        Fourier coefficients with a symmetric frequency coordinate layout
    """
    x_asymm = torch.fft.ifftshift(x, dim=dims)
    x_ft = fftn_asymm(x_asymm, dims=dims, use_rfft=use_rfft)
    x_ft_symm = torch.fft.fftshift(x_ft, dim=dims)
    return x_ft_symm
