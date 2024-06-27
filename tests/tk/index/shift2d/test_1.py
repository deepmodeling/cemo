from cemo.tk.index import shift2d
from cemo.tk.asserto import assert_mat_eq
from cemo.tk.plot import plot_mats
import torch
import os


def test():
    output_dir = "tmp/test1"
    os.makedirs(output_dir, exist_ok=True)
    len_x, len_y = (4, 4)
    x = torch.rand((len_x, len_y))
    shift = torch.tensor([1, 1])
    y = shift2d(x, shift)
    len_x, len_y = x.shape[-2:]
    dx, dy = shift.to(torch.int64)
    x_idx = (torch.arange(len_x) - dx) % len_x
    y_idx = (torch.arange(len_y) - dy) % len_y
    y_expect = x[..., x_idx, :][..., :, y_idx]
    fig, axes = plot_mats(
        [x, y, y_expect],
        ["x", "y", "y_expect"]
    )
    fig.savefig(f"{output_dir}/test1.png")
    assert_mat_eq(y, y_expect)
