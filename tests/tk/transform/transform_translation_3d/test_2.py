from cemo.tk.transform import transform_translation_3d
import torch
from scipy.spatial.transform import Rotation
import numpy


def normalize(x: numpy.ndarray) -> numpy.ndarray:
    return x / numpy.linalg.norm(x)


def test():
    angle = numpy.pi * 0.5
    axis = normalize(numpy.array([0, 0, 1.0]))
    rotvec = angle * axis
    dtype = torch.float32
    N = 1
    rotmat = torch.tensor(
        Rotation.from_rotvec(rotvec).as_matrix(),
        dtype=dtype).expand(N, 3, 3)
    shift_2d = torch.tensor(
        [[0.1, 0.3]],
        dtype=dtype).expand(N, 2)
    x_out = transform_translation_3d(rotmat, shift_2d)
    x_expect = torch.tensor(
        [[[-0.3000],
          [0.1000],
          [0.0000]]],
        dtype=dtype)
    print(rotmat)
    print(x_out.shape)
    print(x_out)
    assert torch.allclose(x_out, x_expect)
