import matplotlib.pyplot as plt
from typing import List, Optional, Union
import numpy
import torch
Tensor = torch.Tensor
Mat = Union[Tensor, numpy.ndarray]
Strs = List[str]


def plot_mats(xs: List[Mat], titles: Optional[Strs] = None):
    fig, axes = plt.subplots(1, len(xs))

    if titles is None:
        titles = [''] * len(xs)

    def aux(i: int):
        raw_x = xs[i]
        if type(raw_x) is Tensor:
            x = raw_x.cpu().numpy()
        else:
            x = raw_x
        axes[i].imshow(x, cmap='viridis')
        axes[i].set_title(titles[i])

    _ = list(map(aux, range(len(xs))))
    return (fig, axes)
