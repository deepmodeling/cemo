"""
Test Butterworth filter.
"""
import torch
import pytest
import itertools
from cemo.tk.mask import make_mask
from cemo.tk.filter import butterworth


@pytest.mark.parametrize(
    (
        "is_rfft",
        "symm",
    ),
    itertools.product(
        [False, True],
        [False, True],
    ),
)
def test_make_mask(
        is_rfft: bool,
        symm: bool,
        ):
    """
    Test Butterworth filter
    """
    print("====================")
    print(f"{is_rfft=}")
    print(f"{symm=}")
    print("====================")
    torch.manual_seed(42)
    shape = "butterworth"
    filter_cutoff = 0.1
    filter_order = 1.0
    L = 6
    size = [L, L]
    ignore_center = False
    if symm and is_rfft:
        with pytest.raises(ValueError):
            mask = make_mask(
                size=size,
                r=filter_cutoff,
                filter_order=filter_order,
                dtype=torch.float32,
                device=torch.device("cpu"),
                shape=shape,
                inclusive=True,
                is_rfft=is_rfft,
                symm=symm,
                ignore_center=ignore_center,
            )
        return
    else:
        mask = make_mask(
                size=size,
                r=filter_cutoff,
                filter_order=filter_order,
                dtype=torch.float32,
                device=torch.device("cpu"),
                shape=shape,
                inclusive=True,
                is_rfft=is_rfft,
                symm=symm,
                ignore_center=ignore_center,
        )

    expect = butterworth(
        size=size,
        cutoff_freq_ratio=filter_cutoff,
        order=filter_order,
        high_pass=False,
        squared_butterworth=True,
        symm=symm,
        is_rfft=is_rfft,
    )

    assert torch.equal(mask, expect)
