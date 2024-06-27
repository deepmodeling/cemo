from torch import Tensor
import torch
from ..t.Atoms import Atoms
# from functorch import vmap
import numba as nb
import numpy as np
from .grid_value import grid_value


@nb.jit(nopython=True)
def genGaussianPoint(positions, radius, weights, gx, gy, gz, res, res_scaling):
    resolution = res * res_scaling
    gridval = 0.0
    dist_cutoff = 15.  # distance cutoff (angstroms)
    cutoff_sqr = dist_cutoff ** 2
    for idx in range(positions.shape[0]):
        sig = radius[idx] * resolution  # / (np.sqrt(2.0) * np.pi)
        r2 = ((positions[idx, 0] - gx) ** 2)
        r2 += (positions[idx, 1] - gy) ** 2
        r2 += (positions[idx, 2] - gz) ** 2
        if r2 < cutoff_sqr:
            gridval += np.math.exp(- r2 / 2.0 / sig ** 2) / np.math.sqrt(2. * np.pi) / sig * weights[idx]
    return gridval


@nb.jit(nopython=True, parallel=True)     
def genGaussianMol(positions, radius, weights, gpoints, res, res_scaling):
    gvals = np.zeros((gpoints.shape[0]))
    for i in nb.prange(gpoints.shape[0]):
        gvals[i] = genGaussianPoint(
            positions, radius, weights,
            gpoints[i,0], gpoints[i,1], gpoints[i,2],
            res, res_scaling)
    return gvals


def atoms2map_numba(
        coords: Tensor,
        radii: Tensor,
        mass: Tensor,
        grid_3d: Tensor,
        res: float,
        res_scaling: float,
        ) -> Tensor:
    """
    Build a simulated 3D cryo-EM map based on atom coordinates
    within a user-defined 3D grid box.

    Args:
        coords: atom coordinates
        radii: atom radii
        mass: atom mass
        grid_3d: coordinates for all the grid points of the output volume 
            of shape(L, L, L, 3) where L is the volume box edge length 
            in pixels.
        res: simulated resolution.
        res_scaling: a scaling factor for the target resolution.
            The sigma of the atomic 3D Gaussian = res * res_scaling

    Returns:
        a 3D volume of shape (L, L, L)
    """
    dtype = grid_3d.dtype
    map_shape = grid_3d.shape[0:-1]  # (L, L, L)
    grid_3d_flat = grid_3d.reshape(-1, 3)  # (L^3, 3)
    N = coords.size(dim=0)  # number of atoms

    gauss = genGaussianMol(
        coords.numpy().astype(np.float32),
        radii.numpy().astype(np.float32), 
        mass.numpy().astype(np.float32),
        grid_3d_flat.numpy().astype(np.float32),
        res=np.float32(res),
        res_scaling=np.float32(res_scaling))
    return torch.tensor(gauss, dtype=dtype).reshape(map_shape)


    # def grid_value(one_grid_point: Tensor) -> Tensor:
    #     """
    #     compute the value for a single grid point

    #     Args:
    #         one_grid_point: (x, y, z) coordinate of a grid point

    #     Returns:
    #         the value at that grid point
    #     """
    #     dist_vec = coords - one_grid_point.expand(N, 3)
    #     dist = torch.norm(dist_vec, p=2, dim=-1)  # (N,)
    #     dist_sqr = torch.square(dist)
    #     sigma = radii * res
    #     amplitude = mass
    #     sigma_sqr = torch.square(sigma)
    #     pi = torch.tensor(torch.pi)
    #     prefactor = amplitude * (1./(sigma * torch.sqrt(2.*pi)))
    #     return torch.sum(prefactor * torch.exp(-dist_sqr/(2. * sigma_sqr)))



def atoms2map_aux(
        coords: Tensor,
        radii: Tensor,
        mass: Tensor,
        grid_3d: Tensor,
        res: Tensor,
        ) -> Tensor:
    """
    Build a simulated 3D cryo-EM map based on atom coordinates
    within a user-defined 3D grid box.

    Args:
        atoms: an instance of the Atoms dataclass
        grid_3d: coordinates for all the grid points of the output volume 
            of shape(L, L, L, 3) where L is the volume box edge length 
            in pixels.
        res: simulated resolution.
        dist_cutoff: distance cutoff for finding atoms near a grid point

    Returns:
        a 3D volume of shape (L, L, L)
    """
    map_shape = grid_3d.shape[0:-1]  # (L, L, L)
    grid_3d_flat = grid_3d.reshape(-1, 3)  # (L^3, 3)
    vol_flat = torch.zeros(grid_3d_flat.size(dim=0))
    
    # fn_grid_value = torch.jit.script(
    #     grid_value,
    #     example_inputs=[
    #         (
    #             grid_3d_flat[0, ...],
    #             coords,
    #             radii,
    #             mass,
    #             torch.tensor(res),
    #         )
    #     ]
    # )

    fn_grid_value = grid_value
    for i in range(vol_flat.size(dim=0)):
        vol_flat[i] = fn_grid_value(
            grid_3d_flat[i, ...],
            coords,
            radii,
            mass,
            res)

    return vol_flat.reshape(map_shape)



# atoms2map = torch.jit.script(
#     atoms2map_aux,
#     [(
#         torch.rand(10, 3, dtype=torch.float32),
#         torch.rand(10, dtype=torch.float32),
#         torch.rand(10, dtype=torch.float32),
#         torch.rand(16, 16, 16, 3, dtype=torch.float32),
#         torch.tensor(5.0),
#     )]
# )
