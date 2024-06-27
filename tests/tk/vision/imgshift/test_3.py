import torch
import pytest
import itertools
from cemo.tk.vision import imgshift
from cemo.tk.math import dht
Tensor = torch.Tensor


# domain="hartley"
@pytest.mark.parametrize(
    (
        "L",
        "indexing",
        "reverse",
        "symm",
    ),
    itertools.product(
        [64],
        ["ij", "xy"],
        [False, True],
        [False, True],
    )
)
def test_imgshift_hartley_domain(
        L: int,
        indexing: str,
        reverse: bool,
        symm: bool,
        ):
    # Setup
    domain = "hartley"
    print("\n===============")
    print(f"{domain=}")
    print(f"{L=}")
    print(f"{indexing=}")
    print(f"{reverse=}")
    print(f"{symm=}")
    print("===============\n")
    image = torch.rand((1, L, L), dtype=torch.float32)

    s = [L, L]  # full-image sizes
    full_image_sizes = torch.tensor(s)
    shift_ratio = torch.tensor([[0.5, 0.5]])
    debug = True

    result = imgshift(
        image,
        s=s,
        shift_ratio=shift_ratio,
        domain=domain,
        is_rfft=False,
        indexing=indexing,
        reverse=reverse,
        symm=symm,
        debug=debug)

    shift_px = shift_ratio * full_image_sizes  # (B, 2)
    if reverse:
        shift_px = -shift_px

    expect = dht.hartley_shift(
        image,
        shift=shift_px,
        dims=[-2, -1],
        indexing=indexing,
        symm=symm,
        debug=debug)

    assert result.shape == image.shape
    assert torch.allclose(result, expect)
