import torch
import numpy
from torch import dtype, Tensor


def read(filename: str, dtype: dtype = torch.float32) -> Tensor:
    return torch.tensor(
        numpy.loadtxt(filename),
        dtype=dtype,
    )
