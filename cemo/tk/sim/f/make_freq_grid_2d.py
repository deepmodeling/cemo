import torch
from torch import Tensor
from typing import Tuple


def is_even(n: int) -> bool:
    return (n % 2 == 0)


def floor_div(x: Tensor, y: float) -> Tensor:
    return torch.div(x, y, rounding_mode="floor")


def make_freq_grid_2d(
        shape: Tuple[int, int],
        apix: float,
        indexing: str,
        dtype: torch.dtype = torch.float32,
        ) -> Tensor:
    """
    Make a 2D frequence grid

    Args:
        shape: shape of the output grid (L, M, N)
        apix: voxel size (unit: angstrom)
        indexing: coordinate indexing order ("xy" or "ij")
        dtype: output data type

    Returns:
        grid_2d: a 2D tensor of shape (N, N)

    Note:
        1. the center pixel (origin) must be at (N//2, N//2) position.
        See: Fig. 1 in "The Electron Microscopy eXchange (EMX) initiative"
        Journal of Structural Biology 194 (2016) 156-163
        http://dx.doi.org/10.1016/j.jsb.2016.02.008
        2. mrcs file stores each image using coordinates in (Y, X) order,
        i.e., axis-0 is the y-axis and axis-1 is the x-axis, which
        corresponds to (indexing="xy").
    """
    n_dim = len(shape)
    N1, N2 = shape
    shape_tensor = torch.tensor([N1, N2], dtype=torch.int32)

    freq_Nyquist = 0.5
    grid_spacing = 1.0 / (floor_div(shape_tensor, 2.) * 2.)
    offset = ((shape_tensor % 2) - 1) * grid_spacing
    L_mins = torch.tensor(-freq_Nyquist, dtype=dtype).repeat(n_dim)
    L_maxs = torch.tensor(freq_Nyquist, dtype=dtype).repeat(n_dim) + offset

    grid_1 = torch.linspace(L_mins[0], L_maxs[0], N1, dtype=dtype)
    grid_2 = torch.linspace(L_mins[1], L_maxs[1], N2, dtype=dtype)
    d1, d2 = torch.meshgrid(grid_1, grid_2, indexing=indexing)
    grid_2d = torch.stack([d1, d2], dim=-1) / apix
    return grid_2d
