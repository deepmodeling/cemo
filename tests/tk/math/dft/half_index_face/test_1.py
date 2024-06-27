import torch
import pytest
from pprint import pprint
import itertools
from typing import Iterable
from cemo.tk.math import dft
from make_expect import make_expect


@pytest.mark.parametrize(
    (
        "batch_size",
        "N",
        "fdims",
        "use_rfft",
    ),
    itertools.product(
        [0, 2],  # batch_size
        [4, 5, 6, 7],  # N
        [[(-3, -2)], [(-3, -2), (-2, -1), (-3, -1)]],   # fdims
        [False, True],  # use_rfft
    )
)
def test(
        batch_size: int,
        N: int,
        fdims: Iterable[int],
        use_rfft: bool,
        ):
    print("==================")
    print(f"N = {N}")
    print(f"fdims = {fdims}")
    print(f"use_rfft = {use_rfft}")
    print("==================")
    # Create an odd shaped tensor
    if batch_size > 0:
        shape = (batch_size, N, N, N)
    else:
        shape = (N, N, N)
    ndim = len(shape)
    dtype = torch.float32
    device = "cpu"
    dims = [-3, -2, -1]
    x = torch.rand(shape, dtype=dtype, device=device)

    x_fft = dft.fftn(
        x,
        dims=dims,
        use_rfft=use_rfft,
        symm_freq=False,
        symm_real=True)
    

    results = dft.half_index_face(x_fft, dims=dims, fdims=fdims)

    expects = make_expect(
        N=N,
        ndim=ndim,
        dims=dims,
        fdims=fdims,
        device=device)

    # pprint(f"result = {results}")
    # pprint(f"expect = {expects}")

    def check(i: int):
        print("------------------")
        print(f"check results[{i}]")
        print(f"sdim = {fdims[i]}")
        result = results[i]
        expect = expects[i]
        print(f"result =\n{result}")
        print(f"expect =\n{expect}")

        def aux(j: int):
            print(f"check result[{j}]")
            if type(expect[j]) is torch.Tensor:
                assert result[j].shape == expect[j].shape, \
                    f"result[{j}].shape = {list(result[j].shape)}, " \
                    f"expect[{j}].shape = {list(expect[j].shape)}"
                torch.testing.assert_close(result[j], expect[j], atol=1e-6, rtol=0.)
            else:
                assert result[j] == expect[j]

        _ = list(map(aux, range(len(expect))))

    _ = list(map(check, range(len(expects))))
