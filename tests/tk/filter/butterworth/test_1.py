import torch
import pytest
import itertools
import numpy as np
from typing import Tuple
from cemo.tk.filter import butterworth
from ref_butterworth import get_nd_butterworth_filter
Tensor = torch.Tensor


@pytest.mark.parametrize(
    (
        "L",
        "cutoff_freq_ratio",
        "order",
        "high_pass",
        "squared_butterworth",
        "symm",
        "is_rfft",
    ),
    itertools.product(
        [3, 4, 5, 6, 32],  # edge size
        [0.005, 0.1, 0.5],  # cutoff_freq_ratio
        [2.0],  # order
        [False, True],  # high_pass
        [False, True],  # squared_butterworth
        [False, True],  # symm
        [False, True],  # is_rfft
    )
)
def test(
        L: Tuple[int],
        cutoff_freq_ratio: float,
        order: float,
        high_pass: bool,
        squared_butterworth: bool,
        symm: bool,
        is_rfft: bool,
        ):
    print("==================")
    print(f"{L=}")
    print(f"size={[L, L]}")
    print(f"{cutoff_freq_ratio=}")
    print(f"{high_pass=}")
    print(f"{squared_butterworth=}")
    print(f"{symm=}")
    print(f"{is_rfft=}")
    print("=================")
    if symm and is_rfft:
        return

    size = [L, L]
    result = butterworth(
            size=size,
            cutoff_freq_ratio=cutoff_freq_ratio,
            order=order,
            high_pass=high_pass,
            squared_butterworth=squared_butterworth,
            symm=symm,
            is_rfft=is_rfft,
            ).cpu().numpy()
    
    expect = get_nd_butterworth_filter(
            shape=size,
            factor=cutoff_freq_ratio,
            order=order,
            high_pass=high_pass,
            real=is_rfft,
            squared_butterworth=squared_butterworth,
            dtype=np.float64,
            )
    
    if symm:
        expect = np.fft.fftshift(expect, axes=(-2, -1))
    
    # print(f"result =\n{result}")
    # print(f"expect =\n{expect}")
    assert np.allclose(result, expect)
