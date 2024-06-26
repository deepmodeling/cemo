import os
import numpy as np
from cemo.tk.mask import dist_mask
from plot_mat import plot_mat
from make_mask import make_mask


# make a square mask
def test():
    test_name = "test_1"
    L = 8
    size = (L, L)
    ord = float("inf")
    r = 2
    inclusive = True

    output_dir = "tmp"
    os.makedirs(output_dir, exist_ok=True)
    f_out = os.path.join(output_dir, f"{test_name}.png")

    mask = dist_mask(size, r=r, ord=ord, inclusive=inclusive).cpu().numpy()
    plot_mat(f_out, mask)
    
    expect = make_mask(size, r=r, ord=ord, inclusive=inclusive)
    # print(f"expect =\n{expect}")
    # print(f"mask =\n{mask}")
    assert np.all(mask == expect)

