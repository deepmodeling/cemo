import torch
import os
import pytest
from run import run
from typing import Tuple, List
Tensor = torch.Tensor


@pytest.mark.parametrize(
        (
            "id",
            "shape",
            "dims",
            "d",
            "shift",
            "indexing",
            "is_rfft",
            "symm",
        ),
        [
            [1, (4, 4), [-2, -1], 1.0, torch.tensor([1., 0.]), "ij", False, False],
            [2, (4, 4), [-2, -1], 1.0, torch.tensor([1., 0.]), "ij", True, False],
            [3, (4, 4), [-2, -1], 1.0, torch.tensor([1., 0.]), "xy", False, False],
            [3, (4, 4), [-2, -1], 1.0, torch.tensor([1., 0.]), "xy", True, False],

            [2, (4, 4), [-2, -1], 1.0, torch.tensor([1., 0.]), "ij", True, True],  # will raise an error
            [1, (4, 4), [-2, -1], 1.0, torch.tensor([1., 0.]), "ij", False, True],
            [3, (4, 4), [-2, -1], 1.0, torch.tensor([1., 0.]), "xy", False, True],


            [1, (10, 4, 4), [-2, -1], 1.0, torch.tensor([1., 2.]).repeat(10, 1), "ij", False, False],
            [2, (10, 4, 4), [-2, -1], 1.0, torch.tensor([2., 1.]).repeat(10, 1), "ij", True, False],
            [3, (10, 4, 4), [-2, -1], 1.0, torch.tensor([1., 2.]).repeat(10, 1), "xy", False, False],
            [3, (10, 4, 4), [-2, -1], 1.0, torch.tensor([2., 1.]).repeat(10, 1), "xy", True, False],

            [1, (10, 4, 4), [-2, -1], 1.0, torch.tensor([1., 2.]).repeat(10, 1), "ij", False, True],
            [2, (10, 4, 4), [-2, -1], 1.0, torch.tensor([2., 1.]).repeat(10, 1), "ij", True, True],
            [3, (10, 4, 4), [-2, -1], 1.0, torch.tensor([1., 2.]).repeat(10, 1), "xy", False, True],
            [3, (10, 4, 4), [-2, -1], 1.0, torch.tensor([2., 1.]).repeat(10, 1), "xy", True, True],
        ]
)
def test(
        id: int,
        shape: Tuple[int],
        dims: List[int],
        d: float,
        shift: Tensor,
        indexing: str,
        is_rfft: bool,
        symm: bool,
        ):
    """
    Test 1
    """
    output_dir = f"tmp/test1-{id}"
    os.makedirs(output_dir, exist_ok=True)
    run(
        output_dir=output_dir,
        shape=shape,
        dims=dims,
        d=d,
        shift=shift,
        indexing=indexing,
        is_rfft=is_rfft,
        symm=symm,
        debug=True,
    )
