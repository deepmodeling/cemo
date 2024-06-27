import torch
from cemo.tk.math.dft import ifft3_symm


def test_ifft3_symm_2d():
    x = torch.rand(3, 3, 3)
    expected_output = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x)))
    assert torch.allclose(ifft3_symm(x), expected_output)
