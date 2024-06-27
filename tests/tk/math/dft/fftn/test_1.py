import torch
import itertools
import pytest
from typing import Tuple
from cemo.tk.math.dft import fftn
Tensor = torch.Tensor


def get_expect(
        x: Tensor,
        dims: Tuple[int],
        symm_freq: bool,
        symm_real: bool,
        use_rfft: bool):
    if symm_freq and use_rfft:
        raise ValueError("symm_freq and use_rfft cannot both be True")

    if symm_real:
        input = torch.fft.ifftshift(x, dim=dims)
    else:
        input = x

    if use_rfft:
        x_ft = torch.fft.rfftn(input, dim=dims)
    else:
        x_ft = torch.fft.fftn(input, dim=dims)

    if symm_freq:
        output = torch.fft.fftshift(x_ft, dim=dims)
    else:
        output = x_ft

    return output


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
    print()
    print("==================")
    print(f"{shape=}")
    print(f"{symm_freq=}")
    print(f"{symm_real=}")
    print(f"{use_rfft=}")
    print(f"{batch_size=}")
    print("==================")
    if batch_size > 0:
        shape = (batch_size, *shape)

    x = torch.rand(shape)

    if batch_size == 0:
        dims = tuple(range(x.ndim))
    else:
        dims = tuple(range(1, x.ndim))
    if symm_freq and use_rfft:
        with pytest.raises(ValueError):
            fftn(
                x,
                dims=dims,
                symm_freq=symm_freq,
                symm_real=symm_real,
                use_rfft=use_rfft)
        return
    result = fftn(
        x,
        dims=dims,
        symm_freq=symm_freq,
        symm_real=symm_real,
        use_rfft=use_rfft)

    expect = get_expect(
        x,
        dims=dims,
        symm_freq=symm_freq,
        symm_real=symm_real,
        use_rfft=use_rfft)

    assert torch.allclose(result, expect, atol=1e-6)
