from time import time
from torch import Tensor
from typing import Tuple
from .atoms2map import atoms2map_numba as fn_atoms2map


def atoms2proj(
        coords: Tensor,
        radii: Tensor,
        mass: Tensor,
        grid_3d: Tensor,
        res: float,
        res_scaling: float,
        axis_order: str = "zyx",
        timeit: bool = False) -> Tuple[Tensor, Tensor]:
    """
    Make a 2D projection image for the input atoms.

    Args:
        coords: atomic coordinates (N, 3)
        radii: atomic radii (N,)
        mass: atomic mass (N,)
        grid_3d: 3d cartensian grid (L, M, N, 3)
            where L, M, and N are the number of voxels 
            for the X, Y, Z dimensions.
        res: target resolution
        res_scaling: resolution scaling factor
        axis_order: output volume and projection order (default: "zyx")
        timeit: whether to pring the timing information
    
    Returns:
        (proj_image, vol)
    """

    t0 = time()
    vol_raw = fn_atoms2map(coords, radii, mass, grid_3d, res, res_scaling)
    t1 = time()

    if axis_order == "zyx":
        # make a projection of the volume along Z
        # note: must use the z-y-x order so that
        # the output projection image is in y-x orientation
        # rather than the x-y orientation.
        # this step is to be consistent with the cryosparc convention.
        vol = vol_raw.transpose(0, 2)
        proj_img = vol.sum(dim=0)
    else:
        vol = vol_raw
        proj_img = vol.sum(dim=2)
    t2 = time()
    
    if timeit:
        t_vol_gen = t1 - t0
        t_cost_proj = t2 - t1
        print(f"time cost (create 3D map): {t_vol_gen*1000:.2f} ms")
        print(f"time cost (projection step): {t_cost_proj*1000:.2f} ms")
    return (proj_img, vol)
