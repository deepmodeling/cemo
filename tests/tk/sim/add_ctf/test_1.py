import matplotlib.pyplot as plt
import numpy as np
from cemo.tk.sim import calc_ctf
import torch
from bin.ctf_utils import plot_ctf, mk_ctf_params, mk_2d_freqs
import numpy as np
from pytest import approx
from torch import Tensor
from cemo.tk.sim import add_ctf, make_freq_grid_2d
import mrcfile


def plot_mat(f_out, X):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    ax1.matshow(X)
    fig.savefig(f_out, bbox_inches="tight")


def read_mrcs(f: str) -> Tensor:
    with mrcfile.open(f, "r") as IN:
        return torch.tensor(IN.data)


def test():
    print("add CTF")
    f_mrcs = "data/one_snr1000_no-ctf_2.mrcs"
    f_img_png = "tmp/one_snr1000_no-ctf_2_ctf.png"
    L = 150
    apix = 1.0
    freq2d = make_freq_grid_2d((L, L), apix=apix, indexing="xy")
    ctf_params = mk_ctf_params()
    ctf = calc_ctf(freq2d, *ctf_params)
    images = read_mrcs(f_mrcs)
    img_w_ctf = add_ctf(images[0], ctf).numpy()
    plot_mat(f_img_png, img_w_ctf)
