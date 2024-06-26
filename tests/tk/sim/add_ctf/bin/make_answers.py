from cryodrgn.ctf import compute_ctf
from ctf_utils import mk_2d_freqs, mk_ctf_params, plot_ctf
import numpy as np


def main():
    L = 128
    apix = 1.0
    f_txt = f"data/L{L}_apix{apix}_ctf.dat"
    f_png = f"data/L{L}_apix{apix}_ctf.png"
    freq2d = mk_2d_freqs(L, apix)
    print(freq2d.shape)
    ctf_params = mk_ctf_params()
    ctf = compute_ctf(freq2d, *ctf_params)
    plot_ctf(f_png, ctf)
    np.savetxt(f_txt, ctf)


main()
