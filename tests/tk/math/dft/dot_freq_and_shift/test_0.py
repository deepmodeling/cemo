import pytest
import torch
import itertools
from typing import Optional, Tuple
from run import run
Tensor = torch.Tensor


# simple 1D shift
@pytest.mark.parametrize(
    (
        "shift",
        "x_shape",
        "dims",
        "indexing",
        "is_rfft",
        "dtype",
        "device",
        "debug",
    ),
    itertools.product(
        [torch.rand(1)],
        [(7,)],
        [(-1,)],
        ["ij", "xy"],
        [False, True],
        [torch.float32, torch.float64, torch.bfloat16],
        [torch.device("cpu"), torch.device(type="cuda")],
        [True],
    )
)
def test_dot_freq_and_shift_shape(
        shift: Tensor,
        x_shape: Tuple[int],
        dims: Tuple[int],
        indexing: str,
        is_rfft: bool,
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
        debug: bool,
        ):
    run(
        shift=shift,
        x_shape=x_shape,
        dims=dims,
        indexing=indexing,
        is_rfft=is_rfft,
        dtype=dtype,
        device=device,
        debug=debug,)
