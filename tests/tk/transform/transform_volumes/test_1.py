from cemo.tk.transform import transform_volumes
import torch
import numpy


def normalize(x: numpy.ndarray) -> numpy.ndarray:
    return x / numpy.linalg.norm(x)


def test():
    L = 3
    N = 1
    dtype = torch.float32
    V = torch.ones([L, L, L], dtype=dtype)
    rotmat = torch.eye(3, dtype=dtype).expand(N, 3, 3)
    x_out = transform_volumes(V, rotmat)
    x_expect = V.expand(N, L, L, L)
    print(x_out)
    print(x_out.shape)
    assert torch.allclose(x_out, x_expect)
