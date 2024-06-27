import pytest
from typing import Tuple
import itertools
import torch
from cemo.tk.math.dft.freqn import freqn


@pytest.mark.parametrize(
    (
        "shape",
        "d",
        "is_rfft",
        "indexing",
        "symm",
        "reverse",
        "dtype",
        "device",
    ),
    itertools.product(
        [
         (1,),
         (2,), (3,),
         (3, 3), (4, 4), (3, 4), (4, 3), (3, 5), (6, 4),
         (3, 3, 3), (4, 4, 4), (3, 4, 5), (4, 3, 5), (3, 5, 6), (6, 4, 5),
         (3, 4, 5, 6, 7),
         (1, 2, 3, 4, 5, 6, 7),
         [3, 3, 4, 5, 6, 7],  # test using list instead of tuple
        ],
        [1.0, 2.0, 3.3],  # d
        [False, True],  # is_rfft
        ["ij", "xy"],
        [False, True],  # symm
        [False, True],  # reverse
        [torch.float32, torch.float64],
        [torch.device("cpu"), torch.device(type="cuda")],
    )
)
def test_freqn_shape(
        shape: Tuple[int, ...],
        d: float,
        is_rfft: bool,
        indexing: str,
        symm: bool,
        reverse: bool,
        dtype: torch.dtype,
        device: torch.device):
    print("========================")
    print(f"shape: {shape}")
    print(f"d: {d}")
    print(f"is_rfft: {is_rfft}")
    print(f"indexing: {indexing}")
    print(f"symm: {symm}")
    print(f"reverse: {reverse}")
    print(f"dtype: {dtype}")
    print(f"device: {device}")
    print("========================")
    shape = tuple(shape)
    ndim = len(shape)

    if is_rfft and symm:
        # is_rfft=True and symm=True cannot be use simutaneously
        with pytest.raises(ValueError):
            result = freqn(
                shape,
                d=d,
                indexing=indexing,
                is_rfft=is_rfft,
                symm=symm,
                reverse=reverse,
                dtype=dtype,
                device=device,
                )
        return
    else:
        result = freqn(
            shape,
            d=d,
            indexing=indexing,
            is_rfft=is_rfft,
            symm=symm,
            reverse=reverse,
            dtype=dtype,
            device=device,
            )
    if is_rfft and indexing == "ij":
        expected_shape = shape[:-1] + (shape[-1]//2 + 1, ndim)
    elif is_rfft and indexing == "xy" and len(shape) == 1:
        expected_shape = (shape[0]//2 + 1,) + (ndim,)
    elif is_rfft and indexing == "xy" and len(shape) == 2:
        expected_shape = (shape[0],) + (shape[1]//2 + 1,) + shape[2:] + (ndim,)
    elif is_rfft and indexing == "xy" and len(shape) > 2:
        expected_shape = shape[:-1] + (shape[-1]//2 + 1, ndim)
    elif indexing == "xy" and len(shape) == 1:
        expected_shape = (shape[0], ndim)
    elif indexing == "xy" and len(shape) > 1:
        expected_shape = shape + (ndim,)
    else:
        expected_shape = shape + (ndim,)

    assert result.shape == expected_shape, \
        f"indexing {indexing}, is_rfft {is_rfft}, ndim {ndim}," \
        f"result.shape: {result.shape}, expected_shape: {expected_shape}"
    assert result.dtype == dtype
    assert result.device.type == device.type

    if indexing == "xy" and ndim >= 2:
        # swap the first two dimensions to ensure the output shape
        # stays the same regardless of the indexing option.
        input_shape = (shape[1], shape[0]) + shape[2:]
    else:
        input_shape = shape

    if (is_rfft and indexing == "ij") or \
       (is_rfft and indexing == "xy" and ndim > 2):
        f1 = [torch.fft.fftfreq(
                N, dtype=dtype, device=device) for N in input_shape[:-1]]
        f2 = [torch.fft.rfftfreq(input_shape[-1], dtype=dtype, device=device)]
        freqs = f1 + f2
    elif is_rfft and indexing == "xy" and ndim <= 2:
        f1 = [torch.fft.rfftfreq(input_shape[0], dtype=dtype, device=device)]
        f2 = [torch.fft.fftfreq(
                N, dtype=dtype, device=device) for N in input_shape[1:]]
        freqs = f1 + f2
    else:
        freqs = [torch.fft.fftfreq(
                    N, dtype=dtype, device=device) for N in input_shape]

    expect = torch.stack(
        torch.meshgrid(
            freqs,
            indexing=indexing
            ),
        dim=-1,
        ) / d
    if symm:
        print(f"shape of expect: {expect.shape}")
        # note: the output has ndim+1 dimensions, where the last dimension is
        # the xyz coordinates.
        # fftshift only needs to be applied to all but the last dimension.
        dims = tuple(range(ndim))
        expect = torch.fft.fftshift(expect, dim=dims)

    if reverse:
        expect = expect.flip(dims=(-1,))
    assert torch.allclose(result, expect, atol=1e-6, rtol=0.0)
