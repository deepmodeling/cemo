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
    dtype = torch.float32
    V = torch.ones([L, L, L], dtype=dtype)
    
    rotmat1 = torch.eye(3, dtype=dtype).expand(1, 3, 3)

    angle = numpy.pi * 0.25
    axis = normalize(numpy.array([1., 2., 1.]))
    rotvec = angle * axis
    rotmat2 = torch.tensor(
        Rotation.from_rotvec(rotvec).as_matrix(),
        dtype=dtype).expand(1, 3, 3)
    rotmat = torch.cat([rotmat1, rotmat2], dim=0)

    shift_2d = torch.tensor(
        [[0., 0.], [0.2, 0.3]],
        dtype=dtype)
    shift_3d = transform_translation_3d(rotmat, shift_2d)
    x_out = transform_volumes(V, rotmat, shift_3d)

    x_expect1 = V.expand(1, L, L, L)
    x_expect2 = torch.tensor(
        [[[1.0000, 0.8883, 0.3598],
          [0.8995, 0.7650, 0.2826],
          [0.1764, 0.0000, 0.0000]],

         [[1.0000, 1.0000, 0.7714],
          [1.0000, 0.9561, 0.5484],
          [0.4400, 0.0537, 0.0000]],

         [[1.0000, 0.9012, 0.1453],
          [0.6850, 1.0000, 0.2559],
          [0.1885, 0.2025, 0.0000]]],
        dtype=dtype).expand(1, L, L, L)
    print("x_expect2 shape:", x_expect2.shape)
    x_expect = torch.cat([x_expect1, x_expect2], dim=0)
    print("x_out", x_out)
    print("x_expect shape:", x_expect.shape)
    assert torch.allclose(x_out, x_expect, atol=1e-4)
