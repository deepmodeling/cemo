import os
import numpy as np
import torch
from plot_mat import plot_mat
from make_mask import make_mask
from cemo.tk.mask import square_mask





# inclusive = False and set device/dtype
def test():
    test_name = "test_2"
    L = 64
    size = (L, L)
    r = 16
    inclusive = False
    device = "cuda:0"
    dtype = torch.float32

    output_dir = "tmp"
    os.makedirs(output_dir, exist_ok=True)
    f_out = os.path.join(output_dir, f"{test_name}.png")

    mask = square_mask(
        size=size,
        r=r,
        inclusive=inclusive,
        dtype=dtype,
        device=device,
        ).cpu().numpy()

    plot_mat(f_out, mask)

    ord = float("inf")
    expect = make_mask(size, r=r, ord=ord, inclusive=inclusive)
    # print(f"expect =\n{expect}")
    # print(f"mask =\n{mask}")
    assert np.all(mask == expect)
