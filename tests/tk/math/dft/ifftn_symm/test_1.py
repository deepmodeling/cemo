import torch
from cemo.tk.math.dft import ifftn_symm


def test_ifftn_symm_1d():
    x = torch.tensor([0., 1., 2., 3., 4.])
    expected_output = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x)))
    assert torch.allclose(ifftn_symm(x, dims=(0,)), expected_output)


def test_ifftn_symm_2d():
    x = torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]])
    expected_output = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x)))
    assert torch.allclose(ifftn_symm(x, dims=(0, 1)), expected_output)


def test_ifftn_symm_3d():
    x = torch.tensor([[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]])
    expected_output = torch.fft.fftshift(
        torch.fft.ifftn(torch.fft.ifftshift(x)))
    assert torch.allclose(ifftn_symm(x, dims=(0, 1, 2)), expected_output)
