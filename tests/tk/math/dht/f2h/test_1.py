import torch
from cemo.tk.math import dht
from cemo.tk.asserto import assert_mat_eq


def test():
    N = 4
    shape = (N, N)
    x = torch.rand(shape)
    dims = [-2, -1]
    x_fft2 = torch.fft.fft2(x, dim=dims)
    y = dht.f2h(x_fft2)
    y_expect = x_fft2.real - x_fft2.imag
    assert_mat_eq(y, y_expect)
