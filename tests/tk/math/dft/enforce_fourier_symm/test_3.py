import torch
import pytest
import itertools
from typing import Iterable, Tuple, List, Union
from cemo.tk.math import dft
from cemo.tk import index
from pprint import pprint
T2 = Tuple[int, int]
LoI = Union[int, List[int]]


# test "fdims" option for 3D tensors
@pytest.mark.parametrize(
        (
            "N",
            "batch_size",
            "inplace",
            "use_conj2",
            "perturb",
            "symm",
            "use_rfft",
            "sdims",
            "fdims",
        ),
        # itertools.product(
        #     [4], # N
        #     [0], # batch_size
        #     [False], # inplace
        #     [False], # use_conj2
        #     [True], # perturb
        #     [False], # symm
        #     [True], # use_rfft
        #     [[(-3, -2)]], # fdims
        # )
        itertools.product(
            [3, 4, 5, 6],  # N
            [0, 5],  # batch_size
            [False, True],  # inplace
            [False, True],  # use_conj2
            [False, True],  # perturb
            [False, True],  # symm
            [False, True],  # use_rfft
            [[-1], [-2], [-1, -2]],   # sdims
            [[(-3, -2)], [(-3, -2), (-2, -1)]],   # fdims
        )
)
def test(
        N: int,
        batch_size: int,
        inplace: bool,
        use_conj2: bool,
        perturb: bool,
        symm: bool,
        use_rfft: bool,
        sdims: Iterable[int],
        fdims: Iterable[T2],
        ):
    print("="*60)
    print(f"N = {N}")
    print(f"batch_size = {batch_size}")
    print(f"inplace = {inplace}")
    print(f"use_conj2 = {use_conj2}")
    print(f"perturb = {perturb}")
    print(f"symm = {symm}")
    print(f"use_rfft = {use_rfft}")
    print(f"fdims = {fdims}")
    print("="*60)

    if batch_size > 0:
        shape = (batch_size, N, N, N)
    else:
        shape = (N, N, N)

    dims = [-3, -2, -1]
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
        fdims_enforce = index.offset_idx(fdims, offset=-1, neg_only=True)
        sdims_enforce = index.offset_idx(sdims, offset=-1, neg_only=True)
        rfft_dim = -2 if use_rfft else None
    else:
        input = x_ft
        dims_fourier = dims
        fdims_enforce = fdims
        sdims_enforce = sdims
        rfft_dim = -1 if use_rfft else None

    if perturb and sdims is not None:
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
    
    if perturb and fdims is not None:
        # note: it is difficult to do perturbation for a symmetric layout.
        # we will do it for an asymmetric layout and then switch back.
        if symm:
            input_asymm = torch.fft.ifftshift(input, dim=dims_fourier)
        else:
            input_asymm = input

        if use_rfft:
            # remove those fdim pairs that contain rfft_dim
            fdims_perturb = list(filter(lambda pair: rfft_dim not in pair, fdims_enforce))
        else:
            fdims_perturb = fdims_enforce

        ids = dft.half_index_face(
            input_asymm,
            dims=dims_fourier,
            fdims=fdims_perturb)
        
        # filter out the indices
        
        # ids_neg = [index.neg_idx(ids[i], dims=fdims[i]) for i in range(len(ids))]
        input_asymm = dft.perturb_coeffs(
            input_asymm, ids=ids, inplace=False)

        if symm:
            input = torch.fft.fftshift(input_asymm, dim=dims_fourier)
        else:
            input = input_asymm

        print(f"shape of input = {input.shape}")
        if use_conj2:
            input_copy = torch.view_as_complex(input.clone())
        else:
            input_copy = input.clone()

    assert dims_fourier is not None, f"dims_fourier={dims_fourier}"
    result = dft.enforce_fourier_symm(
        input,
        dims=dims_fourier,
        symm=symm,
        use_conj2=use_conj2,
        inplace=inplace,
        rfft_dim=rfft_dim,
        sdims=sdims_enforce,
        fdims=fdims_enforce,
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
        s=(N, N, N),
    ).real

    # print(f"result[..., 0]=\n{result[..., 0]}")
    # print(f"x_ft_copy[..., 0]=\n{x_ft_copy[..., 0]}")
    assert torch.allclose(result, x_ft_copy, atol=1e-4, rtol=0.)
    assert torch.allclose(x, x_recon, atol=1e-5, rtol=0.)
