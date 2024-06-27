import torch
from typing import Tuple, List, Union, Iterable, Optional
from cemo.tk.math.dft.ifftn_symm import ifftn_symm
from cemo.tk.math.dft.ifftn_asymm import ifftn_asymm
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def ifftn(
        x: Tensor,
        dims: TuLi,
        symm_freq: bool,
        symm_real: bool = True,
        use_rfft: bool = False,
        s: Optional[Iterable[int]] = None,
        ) -> Tensor:
    """
    N-dimensional inverse fast Fourier transform.
    Note: use_freq and symm_freq cannot both be True.

    Args:
        x: input tensor in Fourier space
        dims: dimensions to be transformed
        symm_freq: whether the input is for a symmetric frequency layout
        symm_real: whether the output is in a symmetric real-space layout
        use_rfft: whether to use rfft

    Returns:
        Real space x with a real-space coordinate layout
    """
    if symm_freq and use_rfft:
        raise ValueError("symm_freq and use_rfft cannot both be True")

    if symm_freq:
        # switch to an asymmetric layout
        input = torch.fft.ifftshift(x, dim=dims)
    else:
        # keep the asymmetric layout
        input = x

    x_asymm = ifftn_asymm(input, dims=dims, use_rfft=use_rfft, s=s)

    if symm_real:
        # switch to a symmetric layout
        output = torch.fft.fftshift(x_asymm, dim=dims)
    else:
        # keep the asymmetric layout
        output = x_asymm

    return output
