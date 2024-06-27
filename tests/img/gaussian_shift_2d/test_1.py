import torch
import numpy
from cemo.img import gaussian_shift_2d
from cemo.io import cs, mrc
from cemo.img import EM_Image
from pytest import approx


def test():
    torch.manual_seed(1020)

    N = 1
    img_size = 5

    images = numpy.ones(
        (N, img_size, img_size),
        dtype=numpy.float32)

    params = numpy.repeat(
            numpy.array(
                [(2.0, [img_size, img_size], [0.0, 0.0])],
                dtype=[
                    ('blob/psize_A', '<f4'),
                    ('blob/shape', '<u4', (2,)),
                    ('alignments3D/shift', '<f4', (2,)),
                ]
            ),
            N,
            axis=0)

    x_in = EM_Image(mrc.MRCX(data=images), cs.CryoSparcCS(data=params))
    shift_bound = torch.tensor([0.02, 0.05]).expand(N, 2)
    std_percent = torch.tensor([0.01, 0.01]).expand(N, 2) * img_size
    print("std_percent", std_percent)
    img_out = gaussian_shift_2d(
        x_in, std_percent, shift_bound, align_corners=False)
    x_out = img_out.mrcs.data
    print(torch.tensor(x_out))
    x_expect = torch.tensor(
        [[[0.9000, 1.0000, 1.0000, 1.0000, 1.0000],
         [0.9000, 1.0000, 1.0000, 1.0000, 1.0000],
         [0.9000, 1.0000, 1.0000, 1.0000, 1.0000],
         [0.9000, 1.0000, 1.0000, 1.0000, 1.0000],
         [0.6838, 0.7598, 0.7598, 0.7598, 0.7598]]])

    assert x_out == approx(x_expect, rel=1e-2, abs=1e-4)
