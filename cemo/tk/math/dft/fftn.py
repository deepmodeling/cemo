import torch
from typing import Tuple, List, Union
from cemo.tk.math.dft.fftn_symm import fftn_symm
from cemo.tk.math.dft.fftn_asymm import fftn_asymm
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def fftn(
        x: Tensor,
        dims: TuLi,
        symm_freq: bool,
        symm_real: bool = True,
        use_rfft: bool = False,
        ) -> Tensor:
    """
    N-dimensional fast Fourier transform.
    Note: use_freq and symm_freq cannot both be True.

    Args:
        x: input tensor
        dims: dimensions to be transformed
        symm_freq: whether the output is for a symmetric frequency layout
        symm_real: whether the input x is for a symmetric real-space layout
        use_rfft: whether to use rfft

    Returns:
        Fourier coefficients with a symmetric frequency coordinate layout
    """
    if symm_freq and use_rfft:
        raise ValueError("symm_freq and use_rfft cannot both be True")

    if symm_real:
        # switch to an asymmetric layout
        input = torch.fft.ifftshift(x, dim=dims)
    else:
        # input is already in an asymmetric layout
        input = x

    x_ft = fftn_asymm(input, dims=dims, use_rfft=use_rfft)

    if symm_freq:
        # switch to a symmetric layout
        output = torch.fft.fftshift(x_ft, dim=dims)
    else:
        # keep the asymmetric layout
        output = x_ft

    return output
