import torch
from typing import List, Union, Iterable, Tuple
from cemo.tk import index
Tensor = torch.Tensor
LTS = List[Union[Tensor, slice]]
T2 = Tuple[int, int]


def make_expect(
        N: int,
        ndim: int,
        dims: Iterable[int],
        fdims: Iterable[T2],
        device: str) -> List[LTS]:

    dims = index.std_idx(dims, ndim)
    fdims = index.std_idx(fdims, ndim)
    i_mid = N // 2

    if N % 2 == 0:
        idx_edge = torch.tensor([0, i_mid], device=device)
        i_end1 = i_mid
        i_beg2 = i_end1 + 1
    else:
        idx_edge = torch.tensor([0], device=device)
        i_end1 = i_mid + 1
        i_beg2 = i_end1

    print(f"idx_edge = {idx_edge}")

    def mk_idx(pair: T2) -> List[LTS]:
        assert len(pair) == 2
        def id_ax(d: int) -> Tensor:
            if d == pair[0]:
                ids1 = torch.arange(1, i_end1, device=device)
                ids2 = torch.arange(i_beg2, N, device=device)
                return torch.cat([ids1, ids2], dim=0)
            elif d == pair[1]:
                return torch.arange(1, i_end1, device=device)
            else:
                return idx_edge

        idx_list = list(map(id_ax, dims))
        ids = index.b_idx(idx_list)
        output = [slice(None) for i in range(ndim)]

        def aux(i: int) -> LTS:
            d = dims[i]
            output[d] = ids[i]

        _ = list(map(aux, range(len(dims))))
        return output

    return list(map(mk_idx, fdims))
