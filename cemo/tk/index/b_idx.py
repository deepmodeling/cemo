import torch
from typing import List
T1 = torch.Tensor


def b_idx(ids: List[T1]) -> List[T1]:
    """
    Convert a list of 1D tensors/slices to a list of broadcastable indices,
    which will have the same indexing effects as using torch.meshgrid,
    but with less memory usage.

    Args:
        ids: a list of 1D index tensors/slices for each dimension

    Returns:
        A list of broadcastable index tensors.
    """
    ndim = len(ids)
    tensor_dims = list(filter(lambda i: type(ids[i]) == T1, range(ndim)))

    def aux(i: int) -> T1:
        idx = ids[i]
        if type(idx) == T1:
            assert idx.ndim == 1, "ids must be a list of 1D tensors"
            shape = [-1 if d == i else 1 for d in tensor_dims]
            return idx.view(shape)
        elif type(idx) == slice:
            return idx
        else:
            raise ValueError("ids must be a list of 1D tensors or slices")

    return list(map(aux, range(ndim)))
