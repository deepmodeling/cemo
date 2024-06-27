import torch
import pytest
from typing import Union
from cemo.tk.ml.dataset.bop.even_split import even_split
Tensor = torch.Tensor


@pytest.mark.parametrize(
    "dataset, batch_size, lengths, expected_output", 
    [
        # Test case 1: testing with dict of tensors and int lengths
        (
            {'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4, 5, 6])},
            3,
            [1, 1, 1],
            [
                {'a': torch.tensor([1]), 'b': torch.tensor([4])},
                {'a': torch.tensor([2]), 'b': torch.tensor([5])},
                {'a': torch.tensor([3]), 'b': torch.tensor([6])},
            ],
        ),
        
        # Test case 2: testing with nested dict of tensors and int lengths
        (
            {
                'a': {'b': torch.tensor([1, 2, 3]),
                'c': torch.tensor([4, 5, 6])},
                'd': torch.tensor([7, 8, 9])},
                3,
                [1, 1, 1],
                [
                    {'a': {'b': torch.tensor([1]), 'c': torch.tensor([4])}, 'd': torch.tensor([7])},
                    {'a': {'b': torch.tensor([2]), 'c': torch.tensor([5])}, 'd': torch.tensor([8])},
                    {'a': {'b': torch.tensor([3]), 'c': torch.tensor([6])}, 'd': torch.tensor([9])},
                ]
        ),
        
        # Test case 3: testing with dict of tensors and float lengths
        (
            {
                'a': torch.tensor([1, 2, 3, 11, 12, 13]),
                'b': torch.tensor([4, 5, 6, 14, 15, 16]),
            },
            6,
            [0.1, 0.3, 0.6],
            [
                {'a': torch.tensor([1]), 'b': torch.tensor([4])},
                {'a': torch.tensor([2, 3]), 'b': torch.tensor([5, 6])},
                {'a': torch.tensor([11, 12, 13]), 'b': torch.tensor([14, 15, 16])},
            ],
        ),
        
        # Test case 4: testing with nested dict of tensors and float lengths
        (
            {'a': {'b': torch.tensor([1, 2, 3]), 'c': torch.tensor([4, 5, 6])}, 'd': torch.tensor([7, 8, 9])},
            3,
            [0.3, 0.3, 0.4],
            [
                {'a': {'b': torch.tensor([1]), 'c': torch.tensor([4])}, 'd': torch.tensor([7])},
                {'a': {'b': torch.tensor([2]), 'c': torch.tensor([5])}, 'd': torch.tensor([8])},
                {'a': {'b': torch.tensor([3]), 'c': torch.tensor([6])}, 'd': torch.tensor([9])},
            ],
        ),
    ]
)
def test_even_split(dataset, batch_size, lengths, expected_output):
    result = even_split(dataset, batch_size, lengths)
    print("------------------")
    print(f"result = {result}")
    print(f"expected_output = {expected_output}")
    check_close(result, expected_output)


def check_close(x: Union[dict, Tensor], y: Union[dict, Tensor]):
    if type(x) is list:
        _ = list(map(lambda i: check_close(x[i], y[i]), range(len(x))))
    elif type(x) is dict:
        _ = list(map(lambda k: check_close(x[k], y[k]), x.keys()))
    elif type(x) is torch.Tensor:
        torch.testing.assert_close(x, y)
    else:
        raise ValueError(f"unsupported type: {type(x)}")
