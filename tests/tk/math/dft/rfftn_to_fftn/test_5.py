from cemo.tk.math import dft
import torch


# batched & enforce=True
def test():
    symm = False
    inplace = False
    enforce = True
    batch_size = 10
    N = 4
    i_mid = N//2  # middle index
    shape = (batch_size, N, N)
    x = torch.rand(shape)
    dims = [-2, -1]
    x_rfft2 = torch.fft.rfft2(x, dim=dims)  # (B, N, N//2+1)

    # do some mutations to the input data
    # to effectively test the enforce=True option
    # indices for the columns where Fourier coefficient symmetry is enforced
    ids_enforce = [0, i_mid]
    x_rfft2[:, -1, ids_enforce] = torch.rand(
        (batch_size, 2), dtype=x_rfft2.dtype)

    assert x_rfft2.shape == (batch_size, N, N//2+1)

    x_fft2 = dft.rfftn_to_fftn(
        x_rfft2, dims=dims, enforce=enforce,
        symm=symm, inplace=inplace)
    y = torch.fft.ifftn(x_fft2, dim=dims).real
    y_expect = x

    torch.testing.assert_close(y, y_expect, atol=1e-6, rtol=0.)
