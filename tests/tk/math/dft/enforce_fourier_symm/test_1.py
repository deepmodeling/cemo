import torch
import pytest
import itertools
from typing import Iterable, List, Union
from cemo.tk.math import dft
from cemo.tk import index
LoI = Union[int, List[int]]


@pytest.mark.parametrize(
        (
            "N",
            "batch_size",
            "inplace",
            "use_conj2",
            "perturb",
            "symm",
            "sdims",
            "use_rfft",
        ),
        # itertools.product(
        #     [3], # N
        #     [1], # batch_size
        #     [False], # inplace
        #     [False], # use_conj2
        #     [False], # perturb
        #     [False], # symm
        #     [[-2]], # sdims
        #     [True], # use_rfft
        # )
        itertools.product(
            [3, 4, 5, 6],  # N
            [1],  # batch_size
            [False, True],  # inplace
            [False, True],  # use_conj2
            [False, True],  # perturb
            [False, True],  # symm
            [[-1], [-2], [-1, -2]],   # sdims
            [False, True],  # use_rfft
        )
)
def test(
        N: int,
        batch_size: int,
        inplace: bool,
        use_conj2: bool,
        perturb: bool,
        symm: bool,
        sdims: Iterable[int],
        use_rfft: bool,
        ):
    print("="*60)
    print(f"N = {N}")
    print(f"batch_size = {batch_size}")
    print(f"inplace = {inplace}")
    print(f"use_conj2 = {use_conj2}")
    print(f"perturb = {perturb}")
    print(f"symm = {symm}")
    print(f"sdims = {sdims}")
    print(f"use_rfft = {use_rfft}")
    print("="*60)

    if batch_size > 0:
        shape = (batch_size, N, N)
    else:
        shape = (N, N)

    dims = [-2, -1]
    # Create an odd shaped tensor
    dtype = torch.float32
    x = torch.rand(shape, dtype=dtype)

    if symm and use_rfft:
        return

    x_ft = dft.fftn(
        x,
        dims=dims,
        symm_freq=symm,
        symm_real=True,
        use_rfft=use_rfft)
    x_ft_copy = x_ft.clone()

    # Call the function with dims=[0, 1], symm=True, use_conj2=False
    if use_conj2:
        input = torch.view_as_real(x_ft)
        dims_fourier = index.offset_idx(dims, offset=-1, neg_only=True)
        sdims_enforce = index.offset_idx(sdims, offset=-1, neg_only=True)
        rfft_dim = -2 if use_rfft else None
    else:
        input = x_ft
        dims_fourier = dims
        sdims_enforce = sdims
        rfft_dim = -1 if use_rfft else None

    if perturb:
        # note: it is difficult to do perturbation for a symmetric layout.
        # we will do it for an asymmetric layout and then switch back.
        sdims_perturb = list(filter(lambda i: i != rfft_dim, sdims_enforce))
        print(f"{sdims_perturb=}")
        if symm:
            input_asymm = torch.fft.ifftshift(input, dim=dims_fourier)
        else:
            input_asymm = input

        ids = dft.half_index_edge(
            input_asymm,
            dims=dims_fourier,
            symm=False,
            sdims=sdims_perturb)

        input_asymm = dft.perturb_coeffs(
            input_asymm, ids=ids, inplace=False)

        if symm:
            input = torch.fft.fftshift(input_asymm, dim=dims_fourier)
        else:
            input = input_asymm

    result = dft.enforce_fourier_symm(
        input,
        dims=dims_fourier,
        sdims=sdims_enforce,
        symm=symm,
        use_conj2=use_conj2,
        inplace=inplace,
        rfft_dim=rfft_dim,
        debug=False,
        )

    if use_conj2:
        result = torch.view_as_complex(result)

    if not inplace:
        assert torch.allclose(x_ft, x_ft_copy, atol=1e-6, rtol=0.)

    x_recon = dft.ifftn(
        result,
        dims=dims,
        symm_freq=symm,
        symm_real=True,
        use_rfft=use_rfft,
        s=(N, N),
    ).real

    # print(f"result=\n{result}")
    # print(f"x_ft_copy=\n{x_ft_copy}")
    assert torch.allclose(result, x_ft_copy, atol=1e-4, rtol=0.)
    assert torch.allclose(x, x_recon, atol=1e-5, rtol=0.)
