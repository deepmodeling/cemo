import torch
from typing import List, Union, Iterable, Tuple
from cemo.tk import index
Tensor = torch.Tensor
LTS = List[Union[Tensor, slice]]
T2 = Tuple[int, int]


def half_index_face(
        x: Tensor,
        dims: Iterable[int],
        fdims: Iterable[T2]) -> List[LTS]:
    """
    Find the indices for the non-edge row/column regions.
    Note: only for non-symmetric layouts.

    Args:
        x: results returned by torch.fft.rfftn
        dims: a tuple of n integers, indicating the dimensions of
            the n-dimensional Fourier transform.
        fdims: a list of face-dimension pairs (must be a 2-tuple),
            i.e., [pair1, pair2, ...].
            For each pair,
            dimension pair[0] of the output will have all non-edge indices.
            dimension pair[1] of the output will have half of the non-edge indices.

    Returns:
        A list of index lists, i.e, [idx1, idx2, ...],
        where each index is a Tensor or slice.
    """
    device = x.device
    ndim = x.ndim  # number of dimensions in the input tensor
    ndim_ft = len(dims)  # number of dimensions in the Fourier transform
    sizes = [x.size(d) for d in dims]
    N = max(sizes)

    # convert indices in fdims to canonical positive indices
    fdims = index.std_idx(fdims, ndim=ndim)
    dims = index.std_idx(dims, ndim=ndim)

    i_mid = N // 2
    if N % 2 == 0:  # even N
        edges = torch.tensor([0, i_mid], device=device)
        i_end1 = i_mid
        i_beg2 = i_end1 + 1
    else:  # odd N
        edges = torch.tensor([0], device=device)
        i_end1 = i_mid + 1
        i_beg2 = i_end1

    def get_idx(sdim_pair: T2) -> LTS:
        assert len(sdim_pair) == 2
        def aux(i: int) -> Tensor:
            d = dims[i]
            if d == sdim_pair[0]:
                ids1 = torch.arange(1, i_end1, device=device)
                ids2 = torch.arange(i_beg2, N, device=device)
                return torch.cat([ids1, ids2], dim=0)
            elif d == sdim_pair[1]:
                return torch.arange(1, i_end1, device=device)
            else:
                return edges

        # make broadcastable indices
        idx = index.b_idx([aux(i) for i in range(ndim_ft)])
        return index.make_idx(ndim=x.ndim, dims=dims, idx=idx)

    return [get_idx(p) for p in fdims]
