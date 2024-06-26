import torch
from typing import List
from functools import reduce
from cemo.tk.index.ix import ix
Tensor = torch.Tensor


def add(xs: List[Tensor]) -> Tensor:
    def aux(x: Tensor, y: Tensor) -> Tensor:
        return x + y
    return reduce(aux, xs)


def test_ix_single_tensor():
    a = torch.tensor([0, 1, 2])
    result = ix([a])
    assert len(result) == 1
    assert result[0].shape == (3,)


def test_ix_2_tensors():
    a = torch.tensor([0, 1, 2])
    b = torch.tensor([3, 4, 5])
    result = ix([a, b])
    assert len(result) == 2
    assert result[0].shape == (3, 1)
    assert result[1].shape == (1, 3)
    assert add(result).shape == (3, 3)


def test_ix_3_tensors():
    a = torch.tensor([0, 1, 2])
    b = torch.tensor([3, 4, 5, 6])
    c = torch.tensor([0, 1])
    result = ix([a, b, c])
    assert len(result) == 3
    assert result[0].shape == (3, 1, 1)
    assert result[1].shape == (1, 4, 1)
    assert result[2].shape == (1, 1, 2)
    assert add(result).shape == (3, 4, 2)


def test_ix_empty_list():
    result = ix([])
    assert len(result) == 0


def test_ix_zero_length_tensor():
    a = torch.tensor([])
    result = ix([a])
    assert len(result) == 1
    assert result[0].shape == (0,)





