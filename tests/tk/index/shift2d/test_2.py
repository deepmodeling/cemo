from cemo.tk.index import shift2d
from cemo.tk.asserto import assert_mat_eq
from cemo.tk.plot import plot_mats
import torch
import os


def test():
    test_id = 2
    output_dir = f"tmp/test{test_id}"
    os.makedirs(output_dir, exist_ok=True)
    len_x, len_y = (4, 4)
    bs = 10
    x = torch.rand((bs, len_x, len_y))
    shift = torch.tensor([1, 1]).expand(bs, -1)
    y = shift2d(x, shift)
    len_x, len_y = x.shape[-2:]
    shift = shift.to(torch.int64)
    dx = shift[..., -2].reshape(-1, 1)
    dy = shift[..., -1].reshape(-1, 1)
    x_range = torch.arange(len_x).expand(bs, -1)
    y_range = torch.arange(len_y).expand(bs, -1)
    x_idx = (x_range - dx) % len_x
    y_idx = (y_range - dy) % len_y
    y_expect = torch.empty_like(x)
    for i in range(bs):
        y_expect[i, :, :] = x[i, x_idx[i, :], :][:, y_idx[i, :]]
    fig, _ = plot_mats(
        [x[-1], y[-1], y_expect[-1]],
        ["x", "y", "y_expect"]
    )
    fig.savefig(f"{output_dir}/test{test_id}.png")
    assert_mat_eq(y, y_expect)
