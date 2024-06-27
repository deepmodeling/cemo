import torch
from cemo.tk.transform import transform_images
from pytest import approx


def test():
    N = 2
    img_size = 5
    x_in = torch.ones(
        (img_size, img_size),
        dtype=torch.float32).expand(N, -1, -1)
    shift_ratio = torch.tensor([-0.5, -0.5]).expand(N, -1)
    x_out = transform_images(
        x_in,
        shift_ratio=shift_ratio,
        align_corners=False)
    x_expect = torch.tensor(
        [[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.2500, 0.5000, 0.5000],
         [0.0000, 0.0000, 0.5000, 1.0000, 1.0000],
         [0.0000, 0.0000, 0.5000, 1.0000, 1.0000]],

        [[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.0000, 0.2500, 0.5000, 0.5000],
         [0.0000, 0.0000, 0.5000, 1.0000, 1.0000],
         [0.0000, 0.0000, 0.5000, 1.0000, 1.0000]]])
    print(x_out)
    assert x_out == approx(x_expect)
