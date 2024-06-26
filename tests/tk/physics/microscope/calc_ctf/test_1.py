import matplotlib.pyplot as plt
import torch
import numpy as np
import pytest
import itertools
from typing import Optional
from cemo.tk.physics.microscope import calc_ctf
from cemo.tk.math import dft
from bin.ctf_utils import plot_ctf, mk_ctf_params


def plot_mat(f_out, X):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    ax1.matshow(X)
    fig.savefig(f_out, bbox_inches="tight")


@pytest.mark.parametrize(
     (
          "L",
          "batch_size",
          "apix",
     ),
     itertools.product(
          [10, 128],
          [0, 1, 10],
          [1., 2., 3., 10., 50., 100.],
     )
)
def test(L: int, batch_size: int, apix: float):
    print()
    print("="*60)
    print(f"{L=}")
    print(f"{batch_size=}")
    print(f"{apix=}")
    print("="*60)

    f_png = f"tmp/t1_L{L}_apix{apix}_ctf.png"
    f_expect = f"data/L{L}_apix{apix}_ctf.dat"
    ctf_expect = torch.tensor(np.loadtxt(f_expect))
    freq2d = dft.freqn(
        (L, L),
        d=1.,
        reverse=True,
        is_rfft=False,
        symm=True,
        dtype=torch.float64,
    )
    ctf_params = mk_ctf_params(batch_size=batch_size)
    ctf1 = calc_ctf(freq2d, params=ctf_params, apix=apix)

    if batch_size > 1:
        ctf2 = ctf_expect.expand(batch_size, -1, -1)
        plot_ctf(f_png, ctf1[0])
    else:
        ctf2 = ctf_expect
        plot_ctf(f_png, ctf1)

    def check(i: int):
        if batch_size > 1:
            x = ctf1[0][i].numpy()
            y = ctf2[0][i].numpy()
        else:
            x = ctf1[i].numpy()
            y = ctf2[i].numpy()
        print(f"[i={i}]:\nx={x[0]:.15f}\ny={y[0]:.15f}\n")

    _ = [check(i) for i in range(3)]

    torch.testing.assert_close(ctf1, ctf2, rtol=0., atol=1e-6)
