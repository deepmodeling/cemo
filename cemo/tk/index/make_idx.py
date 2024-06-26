import torch
from typing import Tuple, List, Union
Tensor = torch.Tensor
TS = Union[Tensor, slice]


def make_idx(ndim: int, dims: Tuple, idx: Tuple) -> List[TS]:
    """
    Make a list of indexing slices.

    Args:
        ndim: number of dimensions
        dims: a tuple of integers, indicating the dimensions to be indexed
        idx: a tuple of indices for the specified dimensions.
            note: len(dims) must be the same as len(sizes)

    Returns:
        A list of indexing slices.
    """
    assert len(dims) == len(idx), "dims and idx must have the same length."
    output = [slice(None) for i in range(ndim)]

    def aux(i: int):
        dim = dims[i]
        output[dim] = idx[i]

    _ = list(map(aux, range(len(dims))))
    return output
