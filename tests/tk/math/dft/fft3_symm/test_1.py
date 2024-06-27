import torch
from cemo.tk.math.dft import fft3_symm


def test_fft3_symm_2d():
    x = torch.rand(3, 3, 3)
    expected_output = torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x)))
    assert torch.allclose(fft3_symm(x), expected_output)
