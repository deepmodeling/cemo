import torch
from typing import List, Tuple
Tensor = torch.Tensor


def ix(idxs: List[Tensor]) -> Tuple[Tensor]:
    """
    An equivalent of the numpy.ix_ function, which
    changes the 1D tensors into appropriate shapes 
    to make use of the broadcasting rule in advanced indexing.

    Args:
        idxs: a list of 1D pytorch tensors.

    Returns:
        a tuple of updated 1D index tensors of broadcastable shapes.
    """
    ndim = len(idxs)

    def aux(i: int) -> Tensor:
        new_shape = [-1 if j == i else 1 for j in range(ndim)]
        return idxs[i].view(new_shape)

    return tuple(map(aux, range(ndim)))
