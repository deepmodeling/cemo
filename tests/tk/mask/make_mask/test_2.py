import torch
from cemo.tk.mask import make_mask, square_mask


def test_make_mask():
    """
    Test square mask
    """
    torch.manual_seed(42)
    L = 7
    size = [L, L]
    r = 2
    mask = make_mask(
            size=size,
            r=r,
            dtype=torch.float32,
            device=torch.device("cpu"),
            shape="square",
            inclusive=True,
    )
    expect = square_mask(
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
