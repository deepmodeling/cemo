import matplotlib.pyplot as plt
import numpy as np
from cemo.tk.sim import make_freq_grid_2d
import torch
from pprint import pprint
from pytest import approx

def plot_mat(f_out, X):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    ax1.matshow(X)
    fig.savefig(f_out, bbox_inches="tight")


def test():
    N = 4
    apix = 1.
    shape = (N, N)
    grid2d = make_freq_grid_2d(shape, apix, indexing="xy")
    xs = grid2d[:, :, 0]
    ys = grid2d[:, :, 1]
    print("X")
    pprint(xs.numpy())
    print("Y")
    pprint(ys.numpy())
    xs_expect = torch.tensor(
        [[-0.5, -0.25,  0., 0.25],
         [-0.5, -0.25,  0., 0.25],
         [-0.5, -0.25,  0., 0.25],
         [-0.5, -0.25,  0., 0.25]],
        dtype=torch.float32,
    )
    ys_expect = torch.tensor(
        [[-0.5, -0.5, -0.5, -0.5],
         [-0.25, -0.25, -0.25, -0.25],
         [0.,  0.,  0.,  0.],
         [0.25,  0.25,  0.25,  0.25]],
        dtype=torch.float32,
    )

    assert xs == approx(xs_expect)
    assert ys == approx(ys_expect)