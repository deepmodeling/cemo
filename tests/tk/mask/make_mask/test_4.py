import torch
import pytest
import itertools
from cemo.tk.mask import make_mask, round_mask


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
# test is_rfft=True
def test_make_mask(
        is_rfft: bool,
        symm: bool,
        ):
    """
    Test round mask
    """
    print("====================")
    print(f"{is_rfft=}")
    print(f"{symm=}")
    print("====================")
    torch.manual_seed(42)
    L = 6
    size = [L, L]
    r = 2
    ignore_center = False
    if symm and is_rfft:
        with pytest.raises(ValueError):
            mask = make_mask(
                size=size,
                r=r,
                dtype=torch.float32,
                device=torch.device("cpu"),
                shape="round",
                inclusive=True,
                is_rfft=is_rfft,
                symm=symm,
                ignore_center=ignore_center,
            )
        return
    else:
        mask = make_mask(
                size=size,
                r=r,
                dtype=torch.float32,
                device=torch.device("cpu"),
                shape="round",
                inclusive=True,
                is_rfft=is_rfft,
                symm=symm,
                ignore_center=ignore_center,
        )

    expect = round_mask(
        size=size,
        r=r,
        dtype=torch.float32,
        device=torch.device("cpu"),
        inclusive=True,
        ignore_center=ignore_center,
    )

    if not symm:
        expect = torch.fft.ifftshift(expect)

    if is_rfft:
        expect = expect[..., :(L // 2 + 1)]
        mid_idx = L // 2
        expect = expect[..., :(mid_idx + 1)]

    assert torch.equal(mask, expect)
