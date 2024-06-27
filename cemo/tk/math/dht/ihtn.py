import torch
from typing import Tuple, List, Union
from cemo.tk.math.dft.fftn import fftn_asymm
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def ihtn(
        x: Tensor,
        dims: TuLi,
        symm_freq: bool,
        symm_real: bool = True,
        ) -> Tensor:
    """
    Inverse n-dimensional Hartley transform.

    Args:
        x: input Hartley coefficient tensor
        dims: dimensions to be transformed
        symm: whether to use a symmetric freq-grid layout

    Returns:
        inverse Hartley-transformed tensor
    """
    assert len(dims) > 0, "dims must be a non-empty list"

    if symm_freq:
        # switch to an asymmetric layout
        input = torch.fft.ifftshift(x, dim=dims)
    else:
        # keep the asymmetric layout
        input = x

    x_ft = fftn_asymm(input, dims=dims, use_rfft=False)

    if symm_real:
        # switch to a symmetric layout
        output = torch.fft.fftshift(x_ft, dim=dims)
    else:
        # keep the asymmetric layout
        output = x_ft

    norm_factor = 1.0 / torch.prod(torch.tensor(x.shape)[dims])

    return (output.real - output.imag) * norm_factor
