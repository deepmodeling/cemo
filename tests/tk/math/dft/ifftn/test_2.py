import torch
import pytest
import itertools
from typing import Tuple
from cemo.tk.math.dft import ifftn, fftn


# Test using random matrices as inputs
@pytest.mark.parametrize(
    (
        "shape",
        "symm_freq",
        "symm_real",
        "use_rfft",
        "batch_size",
    ),
    itertools.product(
        [(3,), (4, 4), (5, 5, 5)],
        [False, True],  # symm_freq
        [False, True],  # symm_real
        [False, True],  # use_rfft
        [0, 1, 10],
    )
)
def test(
        shape: Tuple[int],
        symm_freq: bool,
        symm_real: bool,
        use_rfft: bool,
        batch_size: int,
        ):
    if batch_size > 0:
        shape = (batch_size, *shape)
    print("==================")
    print(f"{shape=}")
    print(f"{symm_freq=}")
    print(f"{symm_real=}")
    print(f"{use_rfft=}")
    print(f"{batch_size=}")
    print("==================")
    x = torch.rand(shape)
    if batch_size == 0:
        dims = tuple(range(x.ndim))
        s = x.shape
    else:
        dims = tuple(range(1, x.ndim))
        s = x.shape[1:]

    if symm_freq and use_rfft:
        with pytest.raises(ValueError):
            fftn(
                x,
                dims=dims,
                symm_freq=symm_freq,
                symm_real=symm_real,
                use_rfft=use_rfft)
        return

    x_ft = fftn(
        x,
        dims=dims,
        symm_freq=symm_freq,
        symm_real=symm_real,
        use_rfft=use_rfft)

    result = ifftn(
        x_ft,
        dims=dims,
        symm_freq=symm_freq,
        symm_real=symm_real,
        use_rfft=use_rfft,
        s=s).real

    assert torch.allclose(x, result, atol=1e-6)
