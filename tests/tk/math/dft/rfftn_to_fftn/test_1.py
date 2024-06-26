from cemo.tk.math import dft
from cemo.tk.asserto import assert_mat_eq
import torch


def test():
    shape = (4, 4)
    symm = False
    inplace = False
    x = torch.rand(shape)
    dims = [-2, -1]
    x_rfft2 = torch.fft.rfft2(x, dim=dims)
    x_fft2 = dft.rfftn_to_fftn(
        x_rfft2, dims=dims, symm=symm, inplace=inplace)
    y = torch.fft.ifftn(x_fft2, dim=dims).real
    y_expect = x
    torch.testing.assert_close(y, y_expect, atol=1e-6, rtol=0.)
