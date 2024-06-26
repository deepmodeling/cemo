import torch
from cemo.tk.mask import make_mask, round_mask


def test_make_mask():
    """
    Test round mask
    """
    torch.manual_seed(42)
    L = 6
    size = [L, L]
    r = 2
    mask = make_mask(
            size=size,
            r=r,
            dtype=torch.float32,
            device=torch.device("cpu"),
            shape="round",
            inclusive=True,
    )
    expect = round_mask(
        size=size,
        r=r,
        dtype=torch.float32,
        device=torch.device("cpu"),
        inclusive=True,
    )
    print("mask")
    print(mask)
    print("expect")
    print(expect)
    assert torch.equal(mask, expect)
