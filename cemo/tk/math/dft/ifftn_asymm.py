import torch
from typing import Tuple, List, Union, Iterable, Optional
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def ifftn_asymm(
        x_ft: Tensor,
        dims: TuLi,
        use_rfft: bool,
        s: Optional[Iterable[int]] = None,
        ) -> Tensor:
    """
    N-dimensional inverse fast Fourier transform.

    Args:
        x_ft: input tensor
        dims: dimensions to be transformed
        use_rfft: whether to use rfft
        s: full signal sizes along the transformed dimensions

    Returns:
        Fourier coefficients with a asymmetric frequency coordinate layout
    """
    if s is not None:
        s = tuple([x for x in s])
    if use_rfft:
        return torch.fft.irfftn(x_ft, dim=dims, s=s)
    else:
        return torch.fft.ifftn(x_ft, dim=dims)
