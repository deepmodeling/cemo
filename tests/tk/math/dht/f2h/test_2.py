import torch
from cemo.tk.math import dht
from cemo.tk.asserto import assert_mat_eq


def test():
    N = 4
    shape = (N, N)
    x = torch.rand(shape)
    dims = [-2, -1]
    x_fft2 = torch.fft.fft2(x, dim=dims)
    x_ht2 = dht.f2h(x_fft2)
    y = dht.ihtn(x_ht2, dims=dims)
    y_expect = x
    assert_mat_eq(y, y_expect)
