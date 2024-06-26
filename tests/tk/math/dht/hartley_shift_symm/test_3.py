from cemo.tk.math import dht
from cemo.tk import index, asserto, plot
import torch
import os


def test():
    test_id = 3
    output_dir = f"tmp/test_{test_id}"
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 10
    shape = (batch_size, 5, 5)
    x = torch.rand(shape)
    shift = torch.tensor([1., 1.], dtype=x.dtype).expand(batch_size, -1)
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
        [x[-1], y_expect[-1], y[-1]], ["x[-1]", "y_expect[-1]", "y[-1]"])
    fig.savefig(f"{output_dir}/test_{test_id}.png")
    asserto.assert_mat_eq(y, y_expect, tol=1e-6)
