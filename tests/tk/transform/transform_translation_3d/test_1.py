from cemo.tk.transform import transform_translation_3d
import torch
from scipy.spatial.transform import Rotation
import numpy


def normalize(x: numpy.ndarray) -> numpy.ndarray:
    return x / numpy.linalg.norm(x)


def test():
    angle = numpy.pi * 0.3
    axis = normalize(numpy.array([1., 2., 3.]))
    rotvec = angle * axis
    dtype = torch.float32
    N = 3
    rotmat = torch.tensor(
        Rotation.from_rotvec(rotvec).as_matrix(),
        dtype=dtype).expand(N, 3, 3)
    x_out = transform_translation_3d(rotmat)
    x_expect = torch.zeros((N, 3, 1), dtype=dtype)
    print(x_out.shape)
    assert torch.allclose(x_out, x_expect)
