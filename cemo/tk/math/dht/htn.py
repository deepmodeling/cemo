import torch
from typing import Tuple, List, Union
from cemo.tk.math.dft.fftn import fftn_asymm
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def htn(
        x: Tensor,
        dims: TuLi,
        symm_freq: bool,
        symm_real: bool = True,
        ) -> Tensor:
    """
    N-dimensional Hartley transform.

    Args:
        x: input tensor
        dims: dimensions to be transformed
        symm: whether to use a symmetric freq-grid layout

    Returns:
        Hartley coefficient tensor (asymmetric layout)
        with the 0-th coefficent for the DC component.
    """
    assert len(dims) > 0, "dims must be a non-empty list"
    # since ihtn make use of fftn again in ihtn and the pairing of
    # fftshift/ifftshift cannot follow the fftn/ifftn pair,
    # we need to handle symm_freq/symm_real here.
    if symm_real:
        # switch to an asymmetric layout
        input = torch.fft.ifftshift(x, dim=dims)
    else:
        # input is already in an asymmetric layout
        input = x

    x_ft = fftn_asymm(input, dims=dims, use_rfft=False)

    if symm_freq:
        # switch to a symmetric layout
        output = torch.fft.fftshift(x_ft, dim=dims)
    else:
        # keep the asymmetric layout
        output = x_ft

    return (output.real - output.imag)
