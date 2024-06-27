from cemo.tk.sim import atoms2map
from cemo.tk.sim import Atoms
from cemo.io import mrc
import torch
from time import time


def test():
    atom_coords = torch.tensor(
            [[10., 0., 0.],
             [0., 10., 0.]]
        )
    N = atom_coords.size(dim=0)
    atom_radii = torch.tensor([1.]).repeat(N)
    atom_mass = torch.tensor([1.]).repeat(N)
    atoms = Atoms(
        coords=atom_coords,
        radii=atom_radii,
        mass=atom_mass
    )
    L = 128
    res = 1.
    axis = torch.linspace(-50, 50, L)
    xs, ys, zs = torch.meshgrid(axis, axis, axis, indexing="xy")
    grid_3d = torch.stack([xs, ys, zs], dim=-1)
    print(grid_3d.shape)
    start = time()
    vol = atoms2map(atoms, grid_3d, res)
    end = time()
    t_cost = end - start
    f_out = "tmp/t1.mrc"
    print(f"time cost: {t_cost*1000:.2f} ms")
    out_mrc = mrc.MRCX(data=vol.numpy())
    mrc.write(f_out, out_mrc)
    print(f_out)
