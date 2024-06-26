from typing import Optional, Iterable
import torch
from cemo.tk.math.dft.dot_freq_and_shift import dot_freq_and_shift
from make_dot import make_dot
Tensor = torch.Tensor


def run(
        shift: Tensor,
        x_shape: Iterable[int],
        dims: Iterable[int],
        indexing: str,
        is_rfft: bool,
        dtype: Optional[torch.dtype],
        device: Optional[torch.device],
        debug: bool,
        ):
    """
    Run a test for dot_freq_and_shift
    """
    if is_rfft:
        s = tuple(x_shape[i] for i in dims)
        x_shape = list(x_shape)
        i_last = dims[-1]
        x_shape[i_last] = x_shape[i_last]//2 + 1
        x_shape = tuple(x_shape)
    else:
        s = None
        x_shape = tuple(x_shape)

    print("========================")
    print(f"shift: {shift}")
    print(f"x_shape: {x_shape}")
    print(f"s: {s}")
    print(f"dims: {dims}")
    print(f"indexing: {indexing}")
    print(f"is_rfft: {is_rfft}")
    print(f"dtype: {dtype}")
    print(f"device: {device}")
    print(f"debug: {debug}")
    print("========================")

    result = dot_freq_and_shift(
        shift=shift,
        x_shape=x_shape,
        s=s,
        dims=dims,
        indexing=indexing,
        is_rfft=is_rfft,
        dtype=dtype,
        device=device,
        debug=debug)

    dims_std = torch.arange(len(x_shape))[list(dims)]

    expect = make_dot(
        shift=shift,
        x_shape=x_shape,
        s=s,
        dims=dims,
        indexing=indexing,
        is_rfft=is_rfft,
        dtype=dtype,
        device=device,
        )

    def check_shape(i: int):
        if i in dims_std:
            return result.size(i) == x_shape[i]
        else:
            return (result.size(i) == 1 or result.size(i) == x_shape[i])

    for i in range(len(x_shape)):
        assert check_shape(i)

    if is_rfft:
        print(f">>> result.shape = {result.shape}")
    assert result.dtype == dtype
    assert result.device.type == device.type
    assert torch.allclose(result, expect, atol=1e-6, rtol=0.0)
