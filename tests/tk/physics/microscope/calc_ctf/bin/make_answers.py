from cryodrgn.ctf import compute_ctf
from ctf_utils import mk_2d_freqs, mk_ctf_params, plot_ctf
import numpy as np
import torch
import itertools


def make(L: int, apix: float):
    print("=============================")
    print(f"{L=}")
    print(f"{apix=}")
    print("=============================")
    f_txt = f"data/L{L}_apix{apix}_ctf.dat"
    f_png = f"data/L{L}_apix{apix}_ctf.png"
    freq2d = mk_2d_freqs(L, apix)
    print(freq2d.shape)
    ctf_params = mk_ctf_params()
    ctf = compute_ctf(
        freq2d,
        ctf_params.df1_A,
        ctf_params.df2_A,
        torch.rad2deg(ctf_params.df_angle_rad),
        ctf_params.accel_kv,
        ctf_params.cs_mm,
        ctf_params.amp_contrast,
        torch.rad2deg(ctf_params.phase_shift_rad),
        ctf_params.bfactor,
    )
    plot_ctf(f_png, ctf)
    np.savetxt(f_txt, ctf)


def main():
    Ls = [10, 128]
    apixs = [1., 2., 3., 10., 50., 100.]

    _ = list(map(
        lambda xs: make(*xs),
        itertools.product(Ls, apixs)
    ))


main()
