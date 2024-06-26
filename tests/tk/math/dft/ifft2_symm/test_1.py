import torch
from cemo.tk.math.dft import ifft2_symm


def test_ifft2_symm_2d():
    x = torch.rand(3, 3)
    expected_output = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(x)))
    assert torch.allclose(ifft2_symm(x, dims=(-2, -1)), expected_output)
