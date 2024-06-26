import torch
import pytest
from pprint import pprint
import itertools
from typing import Iterable
from cemo.tk.math.dft.half_index_edge import half_index_edge
from make_expect import make_expect


@pytest.mark.parametrize(
    (
        "batch_size",
        "N",
        "symm",
        "sdims",
    ),
    itertools.product(
        [0, 2],  # batch_size
        [4, 5, 6, 7],  # N
        [False, True],  # symm
        [[-2], [-1], [-2, -1]],   # sdims
    )
)
def test_half_index_edge_odd(
        batch_size: int,
        N: int,
        symm: bool,
        sdims: Iterable[int],
        ):
    print("==================")
    print(f"N = {N}")
    print(f"symm = {symm}")
    print(f"sdims = {sdims}")
    print("==================")
    if symm and N % 2 == 1:
        # note: we cannot handle symmetry for odd N
        return
    # Create an odd shaped tensor
    if batch_size > 0:
        shape = (batch_size, N, N)
    else:
        shape = (N, N)
    ndim = len(shape)
    dtype = torch.float32
    device = "cpu"
    dims = [-2, -1]
    x = torch.rand(shape, dtype=dtype, device=device)

    x_fft_raw = torch.fft.fft2(x, dim=dims)
    if symm:
        x_fft = torch.fft.fftshift(x_fft_raw, dim=dims)
    else:
        x_fft = x_fft_raw

    results = half_index_edge(x_fft, dims=dims, symm=symm, sdims=sdims)

    expects = make_expect(
        N=N,
        ndim=ndim,
        dims=dims,
        sdims=sdims,
        symm=symm,
        device=device)

    pprint(f"result = {results}")
    pprint(f"expect = {expects}")

    def check(i: int):
        print(f"sdim = {sdims[i]}")
        result = results[i]
        expect = expects[i]

        def aux(j: int):
            print(f"check result[{j}]")
            if type(expect[j]) is torch.Tensor:
                assert torch.allclose(result[j], expect[j], atol=1e-6, rtol=0.)
            else:
                assert result[j] == expect[j]

        _ = list(map(aux, range(len(expect))))

    _ = list(map(check, range(len(expects))))
