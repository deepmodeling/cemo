import torch
import pytest
from cemo.tk.vision import imgshift
Tensor = torch.Tensor


def test_imgshift_invalid_domain():
    # Setup
    domain = "invalid"
    print(f"\n===============")
    print(f"{domain=}")
    print("===============\n")
    L = 64
    s = [L, L]  # full-image sizes
    image = torch.rand((1, L, L))
    shift_ratio = torch.tensor([[0.5, 0.5]])
    is_rfft = False
    indexing = "ij"
    reverse = False
    debug = False

    # Exercise and Verify
    with pytest.raises(ValueError, match="domain must be one of 'fourier', 'hartley', 'real'"):
        imgshift(
            image,
            shift_ratio=shift_ratio,
            s=s,
            domain=domain,
            is_rfft=is_rfft,
            indexing=indexing,
            reverse=reverse,
            debug=debug)
