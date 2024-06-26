import pytest
import torch
from cemo.tk.collection import dict_apply
Tensor = torch.Tensor


def test_dict_apply_with_tensor():
    tensor = torch.tensor([1, 2, 3])
    sizes = (2,)

    def aux(x: Tensor) -> Tensor:
        return x.repeat(*sizes)

    result = dict_apply(tensor, fn=aux)
    expected = torch.tensor([1, 2, 3, 1, 2, 3])
    assert torch.equal(result, expected)


def test_dict_apply_with_dict():
    d = {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])}
    sizes = (2,)

    def aux(x: Tensor) -> Tensor:
        return x.repeat(*sizes)
    result = dict_apply(d, fn=aux)
    expected = {
        'a': torch.tensor([1, 2, 3, 1, 2, 3]),
        'b': torch.tensor([4, 5, 6, 4, 5, 6])}
    for key in result.keys():
        assert torch.equal(result[key], expected[key])
