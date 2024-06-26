import torch


def make_grid(L: int, apix: float, origin: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    L_max = (L-1)/2. * apix
    L_min = -L_max

    offset = 0.
    grid_x = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[0] + offset
    grid_y = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[1] + offset
    grid_z = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[2] + offset
    xs, ys, zs = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_3d = torch.stack([xs, ys, zs], dim=-1)
    print(grid_3d.shape)
    print(grid_x)
    return grid_3d, L_min, L_max

def main():
    origin = torch.zeros(3)
    L = 3
    dtype = torch.float32
    apix = 1.0
    g, L_min, L_max = make_grid(L, apix, origin, dtype)
    # print(g)
    print(L_min, L_max)

main()