import torch
from typing import Tuple, List
import copy
Tensor = torch.Tensor


def neg_idx(idx: List[Tensor], dims: Tuple) -> List[Tensor]:
    """
    Return the negative indices for the specified dimensions.

    Args:
        idx: a list index tensors
        dims: a tuple of dimensions to operate on

    Returns:
        a new list with indices modified for the specified dimensions.
    """
    output = copy.deepcopy(idx)

    def aux(dim: int):
        assert type(idx[dim]) is not slice, \
            f"idx must not be slice for the specified dimension: {dim}."
        output[dim] = -idx[dim]
    _ = list(map(aux, dims))
    return output
