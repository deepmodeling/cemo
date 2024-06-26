import torch
import os
import pytest
from run import run
from typing import Tuple, List
Tensor = torch.Tensor


@pytest.mark.parametrize(
        (
            "shape",
            "dims",
            "shift",
            "indexing",
            "symm",
        ),
        [
            [(4, 4), [-2, -1], torch.tensor([1., 0.]), "ij", False],
            [(4, 5), [-2, -1], torch.tensor([1., 0.]), "ij", False],
            [(5, 5), [-2, -1], torch.tensor([1., 0.]), "ij", False],
            [(4, 4), [-2, -1], torch.tensor([1., 0.]), "xy", False],
            [(4, 5), [-2, -1], torch.tensor([1., 0.]), "xy", False],
            [(5, 5), [-2, -1], torch.tensor([1., 0.]), "xy", False],

            [(10, 4, 4), [-2, -1], torch.tensor([1., 2.]).repeat(10, 1), "ij", False],
            [(10, 4, 5), [-2, -1], torch.tensor([2., 1.]).repeat(10, 1), "ij", False],
            [(10, 5, 5), [-2, -1], torch.tensor([2., 1.]).repeat(10, 1), "ij", False],
            [(10, 4, 4), [-2, -1], torch.tensor([1., 2.]).repeat(10, 1), "xy", False],
            [(10, 4, 5), [-2, -1], torch.tensor([2., 1.]).repeat(10, 1), "xy", False],
            [(10, 5, 5), [-2, -1], torch.tensor([2., 1.]).repeat(10, 1), "xy", False],

            [(10, 4, 5), [-2, -1], torch.tensor([2., 1.]).repeat(10, 1), "xy", True],
            [(10, 5, 5), [-2, -1], torch.tensor([2., 1.]).repeat(10, 1), "xy", True],
        ]
)
def test(
        shape: Tuple[int],
        dims: List[int],
        shift: Tensor,
        indexing: str,
        symm: bool,
        ):
    """
    Test 1
    """
    output_dir = "tmp/test1"
    os.makedirs(output_dir, exist_ok=True)
    run(
        output_dir=output_dir,
        shape=shape,
        dims=dims,
        shift=shift,
        indexing=indexing,
        symm=symm,
        debug=True,
    )
