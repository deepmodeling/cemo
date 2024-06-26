import torch
from typing import Tuple, List, Union
TuLi = Union[Tuple, List]
Tensor = torch.Tensor


def f2h(x: Tensor) -> Tensor:
    """
    Convert Fourier coefficients to Hartley coefficients.

    Args:
        x: input Fourier coeffiicent tensor

    Returns:
        Hartley coefficient tensor
    """
    is_complex = (
        (x.dtype == torch.complex32) or
        (x.dtype == torch.complex64) or
        (x.dtype == torch.complex128)
    )
    assert is_complex, "input tensor must be complex"
    return (x.real - x.imag)
