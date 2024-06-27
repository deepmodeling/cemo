import torch
from typing import Tuple, List, Union, Optional, Iterable
from cemo.tk.math.dft.ifftn_asymm import ifftn_asymm
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def ifftn_symm(
        x_ft: Tensor,
        dims: TuLi,
        use_rfft: bool,
        s: Optional[Iterable[int]] = None,
        ) -> Tensor:
    """
    N-dimensional symmetric inverse fast Fourier transform.

    Args:
        x_ft: input tensor in Fourier space (symmetric layout)
        dims: dimensions to be transformed
        use_rfft: whether to use rfft

    Returns:
        Real space x with a symmetric real-space coordinate layout
    """
    x_ft_asymm = torch.fft.ifftshift(x_ft, dim=dims)
    x = ifftn_asymm(x_ft_asymm, dims=dims, use_rfft=use_rfft, s=s)
    x_symm = torch.fft.fftshift(x, dim=dims)
    return x_symm
