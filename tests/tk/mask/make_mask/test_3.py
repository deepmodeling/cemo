import torch
from cemo.tk.mask import make_mask, round_mask


# test ignore_center=True
def test_make_mask():
    """
    Test round mask
    """
    torch.manual_seed(42)
    L = 6
    size = [L, L]
    r = 2
    ignore_center = True
    mask = make_mask(
            size=size,
            r=r,
            dtype=torch.float32,
            device=torch.device("cpu"),
            shape="round",
            inclusive=True,
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
    print("mask")
    print(mask)
    print("expect")
    print(expect)
    assert torch.equal(mask, expect)
