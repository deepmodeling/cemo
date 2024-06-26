import os
import numpy as np
from plot_mat import plot_mat
from make_mask import make_mask
from cemo.tk.mask import square_mask


def test():
    test_name = "test_1"
    L = 64
    size = (L, L)
    r = 16
    inclusive = True

    output_dir = "tmp"
    os.makedirs(output_dir, exist_ok=True)
    f_out = os.path.join(output_dir, f"{test_name}.png")

    mask = square_mask(
        size=size,
        r=r,
        inclusive=inclusive,
        ).cpu().numpy()

    plot_mat(f_out, mask)

    ord = float("inf")
    expect = make_mask(size, r=r, ord=ord, inclusive=inclusive)
    # print(f"expect =\n{expect}")
    # print(f"mask =\n{mask}")
    assert np.all(mask == expect)
