import torch
from typing import List, Union
Tensor = torch.Tensor
TS = Union[Tensor, slice]


def perturb_coeffs(
        x: Tensor,
        ids: List[List[TS]],
        inplace: bool) -> Tensor:
    """
    Add some random perturbations to the Fourier coefficient symmetry

    Args:
        x: results returned by torch.fft.rfftn
        dims: a tuple of two integers, indicating the dimensions of
            the n-dimensional Fourier transform.
        symm: whether the input coefficient matrix x's layout is symmetric
        inplace: whether to modify the input tensor in-place.

    Returns:
        A modified tensor x
    """
    assert type(ids) is list

    if not inplace:
        x = x.clone()

    def aux(id: List[TS]):
        x[id] += torch.rand_like(x[id])

    _ = list(map(aux, ids))

    return x
