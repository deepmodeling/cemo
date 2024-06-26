import pytest
import torch
from cemo.tk.index import b_idx


def test_b_idx_with_tensor():
    tensor = torch.tensor([1, 2, 3])
    result = b_idx([tensor])
    assert len(result) == 1
    assert torch.equal(result[0], tensor.view([-1]))


def test_b_idx_with_slice():
    slice_obj = slice(1, 5)
    result = b_idx([slice_obj])
    assert len(result) == 1
    assert result[0] == slice_obj


def test_b_idx_with_invalid_type():
    with pytest.raises(ValueError):
        b_idx([1])


def test_b_idx_with_invalid_dimension():
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(AssertionError):
        b_idx([tensor])


def test_b_idx_with_multiple_tensors():
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])
    result = b_idx([tensor1, tensor2])
    assert len(result) == 2
    assert torch.equal(result[0], tensor1.view([-1, 1]))
    assert torch.equal(result[1], tensor2.view([1, -1]))


def test_b_idx_with_multiple_slices():
    slice1 = slice(1, 5)
    slice2 = slice(6, 10)
    result = b_idx([slice1, slice2])
    assert len(result) == 2
    assert result[0] == slice1
    assert result[1] == slice2


def test_b_idx_with_mixed_types():
    tensor = torch.tensor([1, 2, 3])
    slice_obj = slice(1, 5)
    result = b_idx([tensor, slice_obj])
    assert len(result) == 2
    assert torch.equal(result[0], tensor)
    assert result[1] == slice_obj
