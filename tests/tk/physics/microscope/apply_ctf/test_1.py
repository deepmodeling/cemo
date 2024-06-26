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
        [10, 128],
        [False, True],
        [True, False],
        ["fourier", "hartley", "real"],
        [1.0, 2.2]
    ),
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
    ctf = calc_ctf(freq2d, params=ctf_params, apix=apix)

    images = torch.rand((L, L))
    if is_rfft and domain != "real":
        images = torch.fft.rfftn(images, dim=(-2, -1))

    dims = (-2, -1)
    result = apply_ctf(
        images,
        dims=dims,
        ctf_params=ctf_params,
        freq=freq2d,
        domain=domain,
        symm=symm,
        use_rfft=is_rfft,
        apix=apix,
    )

    if domain == "real":
        images_ft = ctf * dft.fftn(
            images, dims=dims, symm=symm, use_rfft=is_rfft)
        expect = dft.ifftn(
            images_ft, dims=dims, symm=symm, use_rfft=is_rfft).real
    else:
        expect = ctf * images

    torch.testing.assert_close(result, expect, rtol=0., atol=1e-6)
