from cemo.tk.transform import (
    transform_volumes,
    transform_translation_3d,
)
import torch
import numpy
from scipy.spatial.transform import Rotation


def normalize(x: numpy.ndarray) -> numpy.ndarray:
    return x / numpy.linalg.norm(x)


def test():
    L = 3
    N = 1
    dtype = torch.float32
    V = torch.ones([L, L, L], dtype=dtype)
    angle = numpy.pi * 0.25
    axis = normalize(numpy.array([1., 2., 1.]))
    rotvec = angle * axis
    rotmat = torch.tensor(
        Rotation.from_rotvec(rotvec).as_matrix(),
        dtype=dtype).expand(N, 3, 3)
    shift_2d = torch.tensor(
        [[0.2, 0.3]],
        dtype=dtype).expand(N, 2)
    shift_3d = transform_translation_3d(rotmat, shift_2d)
    x_out = transform_volumes(V, rotmat, shift_3d)
    x_expect = torch.tensor(
        [[[[1.0000, 0.8883, 0.3598],
          [0.8995, 0.7650, 0.2826],
          [0.1764, 0.0000, 0.0000]],

         [[1.0000, 1.0000, 0.7714],
          [1.0000, 0.9561, 0.5484],
          [0.4400, 0.0537, 0.0000]],

         [[1.0000, 0.9012, 0.1453],
          [0.6850, 1.0000, 0.2559],
          [0.1885, 0.2025, 0.0000]]]],
        dtype=dtype)
    print(x_out)
    assert torch.allclose(x_out, x_expect, atol=1e-4)
