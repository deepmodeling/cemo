from cemo.tk.plot import plot_mats
import numpy
import torch
from typing import Union
Tensor = torch.Tensor
Mat = Union[Tensor, numpy.ndarray]


def plot_mat(x: Mat, title: str = ""):
    plot_mats([x], [title])
