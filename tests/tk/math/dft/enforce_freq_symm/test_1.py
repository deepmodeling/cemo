import torch
import pytest
import itertools
from typing import Iterable, List, Union
from cemo.tk.math import dft
from cemo.tk import index
LoI = Union[int, List[int]]
Tensor = torch.Tensor


@pytest.mark.parametrize(
        (
            "N",
            "inplace",
            "symm",
            "sdims",
            "use_rfft",
        ),

        # itertools.product(
        #     [4],  # N
        #     [False],  # inplace
        #     [False],  # symm
        #     [[0, 1]],  # sdims
        #     [False],  # use_rfft
        # ),

        itertools.product(
            [4, 5, 6, 7],  # N
            [False, True],  # inplace
            [False, True],  # symm
            [[0, 1]],  # sdims
            [False, True],  # use_rfft
        )
)
def test(
        N: int,
        inplace: bool,
        symm: bool,
        sdims: Iterable[int],
        use_rfft: bool,
        ):
    print("="*60)
    print(f"N = {N}")
    print(f"inplace = {inplace}")
    print(f"symm = {symm}")
    print(f"sdims = {sdims}")
    print(f"use_rfft = {use_rfft}")
    print("="*60)
    debug = True

    if symm and use_rfft:
        return

    shape = (N, N)
    dims = [0, 1]
    # Create an odd shaped tensor
    dtype = torch.float32
    x = dft.freqn(
        shape,
        d=1.0,
        indexing="ij",
        is_rfft=use_rfft,
        symm=symm,
        reverse=False,
        dtype=dtype,
        device="cpu")

    if symm and use_rfft:
        return

    input = x.clone()
    dims_fourier = dims
    sdims_enforce = sdims
    rfft_dim = 1 if use_rfft else None

    result = dft.enforce_freq_symm(
        input,
        dims=dims_fourier,
        sdims=sdims_enforce,
        symm=symm,
        inplace=inplace,
        rfft_dim=rfft_dim,
        debug=debug,
        )

    if symm:
        # revert back to the asymmetric layout
        result_asymm = torch.fft.ifftshift(result, dim=dims)
        input_asymm = torch.fft.ifftshift(input, dim=dims)
    else:
        result_asymm = result
        input_asymm = input

    edge_ids = dft.nyquist_edge_ids(N, symm=False)
    idx_list = dft.half_index_edge(
        input, dims=dims, symm=False, sdims=sdims, edge_ids=edge_ids)

    def check(idx: List[Tensor], sdim: int):
        # Since we only need to enforce the symmetry along the symm-dim (1D),
        # the non-symm-dim indices stay positive (no need to negate).
        # Non-symm-dims = [0, N//2] if N is even, [0] if N is odd.
        # Selected index negation ensures compatibility
        # when input x is the output of rfftn.
        # When N is even and x is the output of rfftn,
        # you can get wrong answers if using neg_idx(idx, dims=dims).
        # In that case, x[..., -N//2, ...] is not x[..., N//2, ...].conj()
        # becuase the non-symm-dim is trucated by rfftn.

        if use_rfft and rfft_dim == sdim:
            # no need to enforce symmetry for the rfft_dim dimension
            return
        else:
            idx_neg = index.neg_idx(idx, dims=[sdim])
            x1 = result_asymm[idx]
            x2 = result_asymm[idx_neg]
            torch.testing.assert_close(x1, -x2)

    print(f"idx_list = {idx_list}")

    print("Note: input has been asymmetrized")
    print(f"input_asymm[..., 0] =\n{input_asymm[..., 0]}")
    print(f"input_asymm[..., 1] =\n{input_asymm[..., 1]}")
    print("-" * 60)
    print(f"result_asymm[..., 0] =\n{result_asymm[..., 0]}")
    print(f"result_asymm[..., 1] =\n{result_asymm[..., 1]}")

    def aux(i: int):
        sdim = sdims[i]
        idx = idx_list[i]
        check(idx=idx, sdim=sdim)

    if N % 2 == 0:
        _ = list(map(aux, range(len(idx_list))))
    else:
        torch.testing.assert_close(result, input)
