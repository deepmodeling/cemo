import torch
from cemo.tk.tensor import create


def test_create_without_fill():
    x = create((2, 2), dtype=torch.float32, device='cpu')
    assert x.shape == (2, 2)
    assert x.dtype == torch.float32
    assert x.device == torch.device('cpu')


def test_create_with_fill_tensor():
    fill = torch.tensor(5.0)
    x = create((2, 2), fill=fill, dtype=torch.float32, device='cpu')
    assert torch.all(x == 5.0)
    assert x.shape == (2, 2)
    assert x.dtype == torch.float32
    assert x.device == torch.device('cpu')
