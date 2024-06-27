import torch
from torch import Tensor
from typing import Tuple


def make_grid_3d(
        shape: Tuple[int, int, int],
        apix: float,
        center: Tensor,
        dtype: torch.dtype = torch.float32,
        indexing: str = "ij") -> Tuple[Tensor, Tensor, Tensor]:
    """
    Make a cartesian 3D mesh grid

    Args:
        shape: shape of the output grid (L, M, N)
        apix: voxel size (unit: angstrom)
        center: center coordinates of the 3D grid (3,)
        dtype: output data type
        indexing: coordinate indexing 

    Returns:
        (grid_3d, L_mins, L_maxs)
        grid_3d: a 3D tensor of shape (N, N, N)
        L_mins: min coordinate tensor along each dimension (3,)
        L_maxs: max coordinate tensor along each dimension (3,)
    """
    N1, N2, N3 = shape
    shape_tensor = torch.tensor([N1, N2, N3], dtype=dtype)
    origin_offset = - torch.div(shape_tensor, 2., rounding_mode="floor") * apix
    L_mins = torch.zeros(3, dtype=dtype) + origin_offset
    L_maxs = L_mins + (shape_tensor - 1) * apix
    grid_1 = torch.linspace(L_mins[0], L_maxs[0], N1, dtype=dtype) + center[0]
    grid_2 = torch.linspace(L_mins[1], L_maxs[1], N2, dtype=dtype) + center[1]
    grid_3 = torch.linspace(L_mins[2], L_maxs[2], N3, dtype=dtype) + center[2]
    d1, d2, d3 = torch.meshgrid(grid_1, grid_2, grid_3, indexing=indexing)
    grid_3d = torch.stack([d1, d2, d3], dim=-1)
    return (grid_3d, L_mins, L_maxs)
