import matplotlib.pyplot as plt
import numpy as np
from cemo.tk.sim import add_gaussian_noise
from cemo.tk.mask import circular_disk_mask
import torch


def plot_mat(f_out, X):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    ax1.matshow(X)
    fig.savefig(f_out, bbox_inches="tight")


def test():
    print("add Gaussian noise")
    f_png = "tmp/t3.png"
    N = 1
    L = 128
    snr = 1
    mask = circular_disk_mask(L)
    imgs = torch.tensor(np.eye(L)).unsqueeze(dim=0).repeat(N, 1, 1)
    print("image shape", imgs.shape)
    imgs_w_noise = add_gaussian_noise(imgs, snr=snr, debug=True, mask=mask)
    plot_mat(f_png, imgs_w_noise[0])

test()
