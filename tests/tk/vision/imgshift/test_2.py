import torch
import pytest
import itertools
from cemo.tk.vision import imgshift
from cemo.tk.math import dft
Tensor = torch.Tensor


# domain="fourier"
@pytest.mark.parametrize(
    (
        "L",
        "is_rfft",
        "indexing",
        "reverse",
    ),
    itertools.product(
        [64],
        [False, True],
        ["ij", "xy"],
        [False, True],
    )
)
def test_imgshift_fourier_domain(
        L: int,
        is_rfft: bool,
        indexing: str,
        reverse: bool,
        ):
    # Setup
    domain = "fourier"
    print("\n===============")
    print(f"{domain=}")
    print(f"{L=}")
    print(f"{is_rfft=}")
    print(f"{indexing=}")
    print(f"{reverse=}")
    print("===============\n")
    image = torch.rand((1, L, L), dtype=torch.complex64)
    if is_rfft:
        mid_idx = L // 2
        image = image[:, :, :(mid_idx+1)]

    s = [L, L]  # full-image sizes
    full_image_sizes = torch.tensor(s)
    shift_ratio = torch.tensor([[0.5, 0.5]])
    debug = True

    result = imgshift(
        image,
        s=s,
        shift_ratio=shift_ratio,
        domain=domain,
        is_rfft=is_rfft,
        indexing=indexing,
        reverse=reverse,
        debug=debug)

    shift_px = shift_ratio * full_image_sizes  # (B, 2)
    if reverse:
        shift_px = -shift_px

    expect = dft.fourier_shift(
        image,
        shift=shift_px,
        dims=[-2, -1],
        s=s,
        is_rfft=is_rfft,
        indexing=indexing,
        debug=debug)

    assert result.shape == image.shape
    assert torch.allclose(result, expect)
