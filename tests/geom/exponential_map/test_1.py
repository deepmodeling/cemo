from cemo.geom import exponential_map
import torch
import numpy
from pytest import approx


def torchify(x):
    return torch.tensor(x, dtype=torch.float32, device='cpu')


def test():
    x_in = torchify([1, 0, 0])
    x_expect = torchify([0, 1, 0])
    v_rotation = torchify([0, 0, 1]) * (numpy.pi / 2)
    R = exponential_map(v_rotation)
    x_out = R @ x_in
    print(f"x_out: {x_out}")
    print(f"x_expect: {x_expect}")
    assert x_out == approx(x_expect)
