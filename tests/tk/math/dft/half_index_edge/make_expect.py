import torch
from typing import List, Union
from cemo.tk import index
Tensor = torch.Tensor
TS = Union[Tensor, slice]


def make_expect(
        N: int,
        ndim: int,
        dims: List[int],
        sdims: List[int],
        symm: bool,
        device: str):

    dims = index.std_idx(dims, ndim)
    sdims = index.std_idx(sdims, ndim)
    i_mid = N // 2

    if symm:
        i_DC = i_mid
    else:
        i_DC = 0

    if N % 2 == 0:
        idx_edge = torch.tensor([0, i_mid], device=device)
    else:
        idx_edge = torch.tensor([i_DC], device=device)

    print(f"idx_edge = {idx_edge}")
    print(f"i_DC = {i_DC}")

    def mk_idx(sdim: int) -> List[TS]:
        def id_ax(d: int, sdim: int) -> Tensor:
            if d == sdim:
                return torch.arange(N//2+1, N, device=device)
            else:
                return idx_edge

        idx_list = list(map(lambda d: id_ax(d, sdim), dims))
        ids = index.b_idx(idx_list)
        output = [slice(None) for i in range(ndim)]

        def aux(i: int) -> TS:
            d = dims[i]
            output[d] = ids[i]

        _ = list(map(aux, range(len(dims))))
        return output

    return list(map(mk_idx, sdims))
