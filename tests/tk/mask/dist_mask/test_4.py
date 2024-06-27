import os
import numpy as np
from cemo.tk.mask import dist_mask
from plot_mat import plot_mat
from make_mask import make_mask


# make a square mask
# ignore_center = True
def test():
    test_name = "test_1"
    L = 8
    size = (L, L)
    ord = float("inf")
    r = 2
    inclusive = True
    ignore_center = True

    output_dir = "tmp"
    os.makedirs(output_dir, exist_ok=True)
    f_out = os.path.join(output_dir, f"{test_name}.png")

    mask = dist_mask(
        size,
        r=r,
        ord=ord,
        inclusive=inclusive,
        ignore_center=ignore_center,
        ).cpu().numpy()

    plot_mat(f_out, mask)

    expect = make_mask(
        size,
        r=r,
        ord=ord,
        inclusive=inclusive,
        ignore_center=ignore_center,
        )
    assert np.all(mask == expect)
