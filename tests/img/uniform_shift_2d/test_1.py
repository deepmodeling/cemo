import torch
import numpy
from cemo.img import uniform_shift_2d
from cemo.io import cs, mrc
from cemo.img import EM_Image
from pytest import approx


def test():
    torch.manual_seed(101)

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
    shift_percent = torch.tensor([0.5, 0.5]).expand(N, 2)
    img_out = uniform_shift_2d(
        x_in, shift_percent, align_corners=False)
    x_out = img_out.mrcs.data
    print(torch.tensor(x_out))
    x_expect = torch.tensor(
        [[[0.0000, 0.3684, 0.7514, 0.7514, 0.7514],
         [0.0000, 0.4902, 1.0000, 1.0000, 1.0000],
         [0.0000, 0.4902, 1.0000, 1.0000, 1.0000],
         [0.0000, 0.4902, 1.0000, 1.0000, 1.0000],
         [0.0000, 0.4902, 1.0000, 1.0000, 1.0000]]])
    assert x_out == approx(x_expect, rel=1e-2, abs=1e-4)
