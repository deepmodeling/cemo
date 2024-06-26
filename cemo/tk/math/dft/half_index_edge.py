import torch
from typing import List, Union, Iterable
from cemo.tk import index
Tensor = torch.Tensor
LTS = List[Union[Tensor, slice]]


def half_index_edge(
        x: Tensor,
        dims: Iterable[int],
        symm: bool,
        sdims: Iterable[int]) -> List[LTS]:
    """
    Find the indices for the lower half of the edge elements with
    Fourier coefficient symmetry.
    If N is even, the edge column indices are [0, N//2], otherwise [i_DC].
    If symm = True, i_DC = N//2, otherwise i_DC = 0.
    The row indices are in  range(N//2+1, N).

    Args:
        x: results returned by torch.fft.rfftn
        dims: a tuple of n integers, indicating the dimensions of
            the n-dimensional Fourier transform.
        symm: whether the input coefficient matrix x's layout is symmetric
        sdims: edge dimensions

    Returns:
        A list of index lists, i.e, [idx1, idx2, ...],
        where each index is a Tensor or slice.
    """
    device = x.device
    ndim = x.ndim
    sizes = [x.size(d) for d in dims]
    N = max(sizes)
    if symm and N % 2 == 1:
        raise ValueError("N must be even if symm is True")

    # convert indices in sdims to canonical positive indices
    sdims = index.std_idx(sdims, ndim=ndim)
    dims = index.std_idx(dims, ndim=ndim)

    i_mid = N // 2
    if symm:
        i_DC = i_mid
    else:
        i_DC = 0

    if N % 2 == 0:
        edges = torch.tensor([0, i_mid], device=device)
    else:
        edges = torch.tensor([i_DC], device=device)

    def get_idx(sdim: int) -> LTS:
        def aux(d: int) -> Tensor:
            if d == sdim:
                # note: don't use arange(1, N//2) (if N is even) or arange(1, N//2+1) if N is odd
                # When symm is True and N is odd, the answer is not easy to get.
                return torch.arange(N//2+1, N, device=device)
            else:
                return edges

        idx = index.b_idx([aux(d) for d in dims])
        return index.make_idx(ndim=x.ndim, dims=dims, idx=idx)

    return [get_idx(sdim) for sdim in sdims]
