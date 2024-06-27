import matplotlib.pyplot as plt
import numpy as np
from cemo.tk.sim import add_gaussian_noise
import torch


def plot_mat(f_out, X):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    ax1.matshow(X)
    fig.savefig(f_out, bbox_inches="tight")


def test():
    print("add Gaussian noise")
    f_png = "tmp/t1.png"
    N = 1
    L = 100
    snr = 1
    imgs = torch.tensor(np.eye(L))
    imgs_w_noise = add_gaussian_noise(imgs, snr=snr)
    plot_mat(f_png, imgs_w_noise)


    
test()
