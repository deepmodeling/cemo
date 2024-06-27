from ..t.Atoms import Atoms
from ..t.GridSpecs import GridSpecs
from torch import Tensor
from .atoms2proj import atoms2proj
from .atoms2map import atoms2map_numba as atoms2map
from .make_grid_3d import make_grid_3d
from typing import Tuple
from torch import Tensor


def project_atoms(
        atoms: Atoms,
        gspecs: GridSpecs,
        R: Tensor,
        res_scaling: float = 0.2,
        as_numpy: bool = False,
        axis_order: str = "zyx",
        timeit: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Make projection images of an atomic system

    Args:
        atoms: an Atoms instance
        gspecs: a GridSpecs instance
        R: a 3x3 rotation matrix R for right matrix multiplication
        res_scaling: a scaling factor for the target resolution
        as_numpy: whether to convert the outputs to numpy ndarrays
        axis_order: order of axes of the output projection images and volume
        timeit: whether to print the time cost

    Returns:
        (projection, volume, L_min, L_max, grid_3d)
        projection has shape: (L, L)
        volume: has shape: (L, L, L)
        L_min: min grid coordinates (3,)
        L_max: max grid coordinates (3,)
        grid_3d: 3D grid (L, L, L, 3)
    """
    dtype = atoms.coords.dtype
    # rotate the molecule
    coords_rotated = atoms.coords @ R

    grid_3d, L_min, L_max, = make_grid_3d(
        gspecs.shape, gspecs.apix, gspecs.center, dtype=dtype)

    proj, vol = atoms2proj(
        coords_rotated, atoms.radii, atoms.mass, grid_3d, gspecs.res, 
        res_scaling=res_scaling,
        axis_order=axis_order,
        timeit=timeit)

    if as_numpy:
        return (
            proj.numpy(), vol.numpy(),
            L_min.numpy(), L_max.numpy(),
            grid_3d.numpy())
    else:
        return (proj, vol, L_min, L_max, grid_3d)
