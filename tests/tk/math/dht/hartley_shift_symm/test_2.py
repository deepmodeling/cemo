from cemo.tk.math import dht
from cemo.tk import index, asserto, plot
import torch
import os


def test():
    test_id = 2
    output_dir = f"tmp/test_{test_id}"
    os.makedirs(output_dir, exist_ok=True)
    shape = (5, 5)
    x = torch.rand(shape)
    shift = torch.tensor([1., 1.], dtype=x.dtype)
    dims = [-2, -1]
    x_ht_symm = dht.htn_symm(x, dims)
    y_ht_symm = dht.hartley_shift_symm(
        x_ht_symm,
        shift,
        dims=dims,
        indexing="ij",
        debug=True)
    y = dht.ihtn_symm(y_ht_symm, dims)
    y_expect = index.shift2d(x, shift)
    fig, _ = plot.plot_mats(
        [x, y_expect, y], ["x", "y_expect", "y"])
    fig.savefig(f"{output_dir}/test_{test_id}.png")
    asserto.assert_mat_eq(y, y_expect)
