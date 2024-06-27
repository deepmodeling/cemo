from cemo.geom import exponential_map
import torch
import numpy
from pytest import approx


def torchify(x):
    return torch.tensor(x, dtype=torch.float64, device='cpu')


def test():
    f_in = "data/rotation_vecs.npy"
    f_expect = "data/rotated_vecs.npy"
    v_rotation = torchify(numpy.load(f_in))
    x = torchify([0, 0, 1.0])
    x_expect = torchify(numpy.load(f_expect))
    R = exponential_map(v_rotation)
    x_out = torch.matmul(R, x)
    print(f"x_out: {x_out}")
    print(f"x_expect: {x_expect}")
    print(torch.norm(x_out - x_expect, dim=-1))
    assert x_out == approx(x_expect)
