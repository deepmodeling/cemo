from cemo.tk.rotation import from_6d_to_rotmat
import torch
import numpy


def normalize(x: numpy.ndarray) -> numpy.ndarray:
    return x / numpy.linalg.norm(x)


def test():
    x_6d = torch.tensor([
        [2.0, 1.0, 3.0, 5.0, 1.1, 6.0],
        [2.0, 0.0, 0.0, 0.0, 1.1, 0.0],
    ])
    x_out = from_6d_to_rotmat(x_6d)
    x_expect = torch.tensor(
        [[[0.5345,  0.2673,  0.8018],
          [0.6420, -0.7454, -0.1795],
          [0.5496,  0.6107, -0.5700]],

         [[1.0000,  0.0000,  0.0000],
          [0.0000,  1.0000,  0.0000],
          [0.0000,  0.0000,  1.0000]]])

    print(x_out)
    assert torch.allclose(x_out, x_expect, atol=1e-4)
