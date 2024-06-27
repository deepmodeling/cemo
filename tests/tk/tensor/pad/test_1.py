import torch
from cemo.tk.tensor import pad


def test_pad_without_fill_1():
    x = torch.tensor([[1, 2], [3, 4]])
    dims = [1]
    new_sizes = [4]
    x_new = pad(x, dims=dims, new_sizes=new_sizes)
    assert x_new.shape == (2, 4)
    assert x_new.dtype == x.dtype
    assert x_new.device == x.device
    assert torch.allclose(x_new[:, :2], x)


def test_pad_without_fill_2():
    x = torch.tensor([[1, 2], [3, 4]])
    dims = [0, 1]
    new_sizes = [4, 4]
    x_new = pad(x, dims=dims, new_sizes=new_sizes)
    assert x_new.shape == (4, 4)
    assert x_new.dtype == x.dtype
    assert x_new.device == x.device


def test_pad_without_fill_3():
    # test negative dims (len(dims) == 1)
    x = torch.rand(4, 3)
    dims = [-1]
    new_sizes = [4]
    x_new = pad(x, dims=dims, new_sizes=new_sizes)
    assert x_new.shape == (4, 4)
    assert x_new.dtype == x.dtype
    assert x_new.device == x.device
    assert torch.allclose(x_new[:, :3], x)


def test_pad_without_fill_4():
    # test negative dims (len(dims) == 1)
    x = torch.rand(4, 3)
    dims = [-2, -1]
    new_sizes = [5, 5]
    x_new = pad(x, dims=dims, new_sizes=new_sizes)
    assert x_new.shape == (5, 5)
    assert x_new.dtype == x.dtype
    assert x_new.device == x.device
    assert torch.allclose(x_new[:4, :3], x)


def test_pad_with_fill_1():
    dtype = torch.float32
    x = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
    dims = [-1]
    new_sizes = [4]
    fill = 1.7
    x_new = pad(x, dims=dims, new_sizes=new_sizes, fill=fill)
    print(f"x_new: \n{x_new}")
    assert x_new.shape == (2, 4)
    assert x_new.dtype == x.dtype
    assert x_new.device == x.device
    assert torch.all(x_new[:, 2:] == fill)


def test_pad_with_fill_2():
    dtype = torch.float32
    x = torch.tensor([[1, 2], [3, 4]], dtype=dtype)
    dims = [-2, -1]
    new_sizes = [4, 4]
    fill = 1.1
    x_new = pad(x, dims=dims, new_sizes=new_sizes, fill=fill)
    print(f"x_new: \n{x_new}")
    assert x_new.shape == (4, 4)
    assert x_new.dtype == x.dtype
    assert x_new.device == x.device
    assert torch.all(x_new[2:] == fill)
    assert torch.all(x_new[:, 2:] == fill)


