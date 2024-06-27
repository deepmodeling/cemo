from cemo.tk.sim import atoms2map
from cemo.tk.sim import Atoms
from cemo.io import mrc, pdb
from cemo.tk import const
from cemo.tk import transform

import torch
from time import time
import numpy as np
import mrcfile
try:
    import openmm.unit as unit
except ImportError: 
    import simtk.unit as unit 


def test():
    f_in = "input/1UBQ_cen.pdb"
    f_out_mrc = "tmp/1ubq_tmp.mrc"
    f_out_mrcs = "tmp/1ubq_proj.mrcs"
    pdb_obj = pdb.read(f_in)
    atom_coords = pdb_obj.coords
    atom_topo = pdb_obj.topo
    element_names = [a.element.symbol for a in pdb_obj.topo.atoms()]
    N = atom_coords.size(dim=0)
    # N = 10
    # atom_radii = torch.tensor([0.1]).repeat(N)
    # atom_mass = torch.tensor([1.]).repeat(N)
    radii = torch.tensor([const.atom_radius(elem) for elem in element_names])
    mass = torch.tensor(
        np.array([a.element.mass.value_in_unit(unit.amu) for a in atom_topo.atoms()])
    )
    center_coords = atom_coords.mean(dim=0)
    print(center_coords)
    atom_coords -= center_coords
    atoms = Atoms(
        coords=atom_coords[:N, :],
        radii=radii,
        mass=mass
    )
    print(f"{N} atoms")
    print(atom_coords.shape)
    print("min", torch.min(torch.abs(atom_coords)))
    print("max", torch.max(torch.abs(atom_coords)))
    # return
    L = 128
    res = 2.0 * 0.2
    apix = 0.5
    L_max = L/2. * apix
    L_min = -L_max
    axis = torch.linspace(L_min, L_max, L)
    xs, ys, zs = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid_3d = torch.stack([xs, ys, zs], dim=-1)
    print(grid_3d.shape)

    start = time()
    vol = atoms2map(atoms, grid_3d, res)
    end = time()
    t_cost = end - start
    
    print(f"time cost: {t_cost*1000:.2f} ms")
    # out_mrc = mrc.MRCX(data=vol.numpy().T.astype(np.float32))
    vol_data = vol.numpy().T.astype(np.float32)
    vol_data_tensor = torch.tensor(vol_data)
    with mrcfile.new(f_out_mrc, overwrite=True) as out_mrc:
        out_mrc.set_data(vol_data)
        out_mrc.voxel_size = apix
        # out_mrc.header.nxstart = L_min
        # out_mrc.header.nystart = L_min
        # out_mrc.header.nzstart = L_min
        out_mrc.header.origin.x = L_min
        out_mrc.header.origin.y = L_min
        out_mrc.header.origin.z = L_min
        # out_mrc.flush()
    # mrc.write(f_out, out_mrc)

    proj_img = transform.project_volumes(vol_data_tensor)
    print(proj_img.shape)
    with mrcfile.new_mmap(f_out_mrcs, shape=(1, L, L), overwrite=True, mrc_mode=2) as OUT2:
        OUT2.set_data(proj_img.numpy())
        OUT2.header.origin.x = L_min
        OUT2.header.origin.y = L_min
        OUT2.header.origin.z = L_min
    print(f_out_mrc)
    print(f_out_mrcs)
 