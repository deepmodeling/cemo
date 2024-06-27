import torch
from typing import Tuple, List, Union
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def fftn_asymm(x: Tensor, dims: TuLi, use_rfft: bool) -> Tensor:
    """
    N-dimensional fast Fourier transform.

    Args:
        x: input tensor
        dims: dimensions to be transformed
        use_rfft: whether to use rfft

    Returns:
        Fourier coefficients with a asymmetric frequency coordinate layout
    """
    if use_rfft:
        return torch.fft.rfftn(x, dim=dims)
    else:
        return torch.fft.fftn(x, dim=dims)
