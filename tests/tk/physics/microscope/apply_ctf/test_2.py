"""
Test apply_ctf's output in real space using ifftn.

YHW 2024.05.08
"""
import matplotlib.pyplot as plt
import torch
import itertools
import pytest
from typing import Tuple, List, Union
import cemo.tk.math.dft as dft
from cemo.tk.physics.microscope import calc_ctf, apply_ctf
from bin.ctf_utils import mk_ctf_params
Tensor = torch.Tensor
TuLi = Union[Tuple, List]


def plot_mat(f_out, X):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    ax1.matshow(X)
    fig.savefig(f_out, bbox_inches="tight")


@pytest.mark.parametrize(
    (
        "L",
        "is_rfft",
        "symm",
        "domain",
        "apix",
    ),
    itertools.product(
        [4, 5, 6, 7],  # L
        [False, True],  # is_rfft
        [False, True],  # symm
        ["fourier", "hartley", "real"],  # domain
        [1.0, 2.2],  # apix
    ),
    # itertools.product(
    #     [4],
    #     [False],  # is_rfft
    #     [False],  # symm
    #     ["fourier"],
    #     [2.2],
    # ),
)
def test(L: int, is_rfft: bool, symm: bool, domain: str, apix: float):
    print()
    print("="*60)
    print(f"{L=}")
    print(f"{is_rfft=}")
    print(f"{symm=}")
    print(f"{domain=}")
    print(f"{apix=}")
    print("="*60)
    shape = (L, L)

    # if L % 2 == 0 and is_rfft:
    #     print("when L is even, cannot use is_rfft=True")
    #     return

    if is_rfft and symm:
        with pytest.raises(ValueError):
            dft.freqn(
                shape,
                d=1.,
                indexing="ij",
                is_rfft=is_rfft,
                symm=symm,
            )
        return
    else:
        freq2d = dft.freqn(
            shape,
            d=1.,
            indexing="ij",
            is_rfft=is_rfft,
            symm=symm,
        )

    ctf_params = mk_ctf_params()
    # ctf_params = None

    if domain == "real":
        images = torch.rand(shape)
        images_full_size = images
    else:
        images_real = torch.rand(shape)
        images = dft.fftn(
            images_real,
            dims=(-2, -1),
            symm_freq=symm,
            symm_real=True,
            use_rfft=is_rfft)
        images_full_size = dft.fftn(
            images_real,
            dims=(-2, -1),
            symm_freq=symm,
            symm_real=True,
            use_rfft=False
        )

    dims = (-2, -1)
    result_tmp = apply_ctf(
        images,
        dims=dims,
        ctf_params=ctf_params,
        freq=freq2d,
        domain=domain,
        symm=symm,
        use_rfft=is_rfft,
        apix=apix,
        enforce_freq_symm=True,
    )

    if domain == "real":
        result_real_raw = result_tmp
        result_real = result_tmp
    else:
        result_real_raw = dft.ifftn(
            result_tmp,
            dims=dims,
            symm_freq=symm,
            symm_real=True,
            s=shape,
            use_rfft=is_rfft)
        result_real = result_real_raw.real

    freq2d_full_size = dft.freqn(
            shape,
            d=1.,
            indexing="ij",
            is_rfft=False,
            symm=symm,
        )

    freq2d_full_size = dft.enforce_freq_symm(
        freq2d_full_size,
        dims=(0, 1),
        sdims=(0, 1),
        symm=symm,
        inplace=False,
        rfft_dim=None,
        N=L,
        debug=True,
    )

    ctf_full_size = calc_ctf(freq2d_full_size, params=ctf_params, apix=apix)

    if domain == "real":
        images_ft_full_size = ctf_full_size * dft.fftn(
            images_full_size,
            dims=dims,
            symm_freq=symm,
            symm_real=True,
            use_rfft=False)
        expect_real = dft.ifftn(
            images_ft_full_size,
            dims=dims,
            symm_freq=symm,
            symm_real=True,
            use_rfft=False).real
    else:
        expect_ft = ctf_full_size * images_full_size
        expect_real_raw = dft.ifftn(
            expect_ft,
            dims=dims,
            symm_freq=symm,
            symm_real=True,
            s=shape,
            use_rfft=False)
        expect_real = expect_real_raw.real
    #     print(f"expect_ft=\n{expect_ft.real.numpy()}+\n{expect_ft.imag.numpy()}j")

    print(f"freq2d[..., 0]=\n{freq2d[..., 0]}")
    print(f"freq2d[..., 1]=\n{freq2d[..., 1]}")
    print(f"freq2d_full_size[..., 0]=\n{freq2d_full_size[..., 0]}")
    print(f"freq2d_full_size[..., 1]=\n{freq2d_full_size[..., 1]}")

    ctf = calc_ctf(freq2d, params=ctf_params, apix=apix)
    print(f"ctf=\n{ctf}")
    print(f"ctf_full_size=\n{ctf_full_size}")

    print(f"size of result_real={result_real.size()}")
    print(f"size of expect_real={expect_real.size()}")

    if not is_rfft and domain != "real":
        sum_result_real_img = torch.sum(result_real_raw.imag)
        sum_expect_real_img = torch.sum(expect_real_raw.imag)
        print(f"result_real_raw's img = {sum_result_real_img}")
        print(f"expect_real's img = {sum_expect_real_img}")
        zero = torch.tensor(0.)
        torch.testing.assert_close(sum_expect_real_img, zero)
        torch.testing.assert_close(sum_result_real_img, zero)
    torch.testing.assert_close(result_real, expect_real, rtol=0, atol=1e-6)
