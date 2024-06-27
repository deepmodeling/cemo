from cemo.tk.rotation import from_6d_to_rotmat
import torch
import numpy


def normalize(x: numpy.ndarray) -> numpy.ndarray:
    return x / numpy.linalg.norm(x)


def test():
    x_6d = torch.tensor([
        [2.0, 0.0, 0.0, 0.0, 1.1, 0.0],
    ])
    x_out = from_6d_to_rotmat(x_6d)
    x_expect = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert torch.allclose(x_out, x_expect)
