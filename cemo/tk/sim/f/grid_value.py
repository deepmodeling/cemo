from torch import Tensor
import torch


@torch.jit.script
def grid_value(
        one_grid_point: Tensor,
        coords: Tensor,
        radii: Tensor,
        mass: Tensor,
        res: Tensor) -> Tensor:
    """
    compute the value for a single grid point

    Args:
        one_grid_point: (x, y, z) coordinate of a grid point

    Returns:
        the value at that grid point
    """
    N = coords.size(dim=0)
    dist_vec = coords - one_grid_point.expand(N, 3)
    dist = torch.norm(dist_vec, p=2, dim=-1)  # (N,)
    ids = dist < 5.0  # find indices for atoms within a cutoff
    dist_sqr = torch.square(dist[ids])
    sigma = radii[ids] * res
    amplitude = mass[ids]
    sigma_sqr = torch.square(sigma)
    pi = torch.tensor(torch.pi)
    prefactor = amplitude * (1./(sigma * torch.sqrt(2.*pi)))
    return torch.sum(prefactor * torch.exp(-dist_sqr/(2. * sigma_sqr)))