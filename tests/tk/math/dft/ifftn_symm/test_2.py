import torch
from cemo.tk.math.dft import ifftn_symm, fftn_symm


def test():
    x = torch.rand(3, 3)
    expected_output = x
    y = ifftn_symm(fftn_symm(x, dims=(0, 1)), dims=(0, 1)).real
    # y = torch.fft.ifft2(torch.fft.fft2(x)).real
    assert torch.allclose(y, expected_output)
