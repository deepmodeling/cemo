import torch
from cemo.tk.math.dft import fft2_symm


def test_fft2_symm_2d():
    x = torch.rand(3, 3)
    expected_output = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x)))
    assert torch.allclose(fft2_symm(x, dims=(-2, -1)), expected_output)
