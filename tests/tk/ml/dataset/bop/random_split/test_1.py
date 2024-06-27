import pytest
import torch
from typing import List, Optional, Union
from cemo.tk.ml.dataset.bop.random_split import random_split
Tensor = torch.Tensor
FoI = Union[float, int]
DoT = Union[dict, Tensor]


@pytest.mark.parametrize(
        (
            "batch_size",
            "input",
            "expect",
            "lengths",
        ),
        [
            # test depth-1 dict
            (
                3,
                {"a": torch.tensor([1, 2, 3]), "b": torch.tensor([4, 5, 6])},
                [
                    {"a": torch.tensor([1, 3]), "b": torch.tensor([4, 6])},
                    {"a": torch.tensor([2]), "b": torch.tensor([5])},
                ],
                [0.5, 0.5],
            ),

            # test depth-2 dict
            (
                3,
                {"a": {"x": torch.tensor([1, 2, 3]), "y": torch.tensor([4, 5, 6])}, "b": {"z": torch.tensor([7, 8, 9])}},
                [
                    {"a": {"x": torch.tensor([1, 3]), "y": torch.tensor([4, 6])}, "b": {"z": torch.tensor([7, 9])}},
                    {"a": {"x": torch.tensor([2]), "y": torch.tensor([5])}, "b": {"z": torch.tensor([8])}},
                ],
                [0.5, 0.5],
            ),

            # test depth-1 2D tensors
            (
                3,
                {
                    "a": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                    "b": torch.tensor([[10, 11, 12], [13, 14, 15], [16, 17, 18]]),
                },
                [
                    {
                        "a": torch.tensor([[1, 2, 3], [7, 8, 9]]),
                        "b": torch.tensor([[10, 11, 12], [16, 17, 18]]),
                    },
                    {
                        "a": torch.tensor([[4, 5, 6]]),
                        "b": torch.tensor([[13, 14, 15]]),
                    },
                ],
                [0.5, 0.5],
            ),
        ],
)
def test_dict_split(
        batch_size: int,
        input: dict,
        expect: List[dict],
        lengths: List[FoI],
        ):
    print("=====================================")
    print(f"{input=}")
    print(f"{expect=}")
    print(f"{lengths=}")
    print("=====================================")
    seed = 42
    output = random_split(
        input,
        batch_size=batch_size,
        lengths=lengths,
        seed=seed,
        debug=True,
    )
    print(f"{output=}")

    def compare(obj: DoT, ref: DoT, n_indent: int = 0):
        indent = " " * n_indent
        if type(ref) is Tensor:
            x = obj[:]
            print(f"{indent}{x=}")
            print(f"{indent}{ref=}")
            assert type(x) is Tensor
            torch.testing.assert_close(x, ref)
        elif type(ref) is dict:
            assert type(obj) is dict
            assert obj.keys() == ref.keys()
            for k in ref.keys():
                print(f"{indent}{k}:")
                compare(obj[k], ref[k], n_indent=n_indent+2)
        else:
            raise ValueError(f"Shouldn't happen!: type(ref) = {type(ref)}")

    for i in range(len(lengths)):
        print("-----------------")
        print(f"i={i}")
        print("-----------------")
        compare(output[i], expect[i])
