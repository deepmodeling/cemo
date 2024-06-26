from cemo.tk.math import dft
from cemo.tk.asserto import assert_mat_eq
import torch


def test():
    symm = False
    inplace = True
    img_size = 4
    shape = (img_size, img_size)
    x = torch.rand(shape)
    dims = [-2, -1]
    x_rfft2 = torch.fft.rfft2(x, dim=dims)
    if img_size % 2 == 0:
        pad = img_size//2 - 1
    else:
        pad = img_size//2
    x_rfft2_padded = torch.nn.functional.pad(x_rfft2, pad=(0, pad))

    _ = dft.rfftn_to_fftn(
        x_rfft2_padded, dims=dims,
        symm=symm, inplace=inplace)
    y = torch.fft.ifftn(x_rfft2_padded, dim=dims).real
    y_expect = x
    torch.testing.assert_close(y, y_expect, atol=1e-6, rtol=0.)
