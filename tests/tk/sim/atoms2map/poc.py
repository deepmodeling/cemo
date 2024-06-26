from re import I
from cemo.tk.sim.f.atoms2map import atoms2map_numba as atoms2map
from cemo.tk.sim import Atoms
from cemo.io import mrc, pdb
from cemo.tk import const
from cemo.tk import transform
from cemo.tk.sim import atoms2proj

import cemo
import os
import mdtraj
from mdtraj.utils import in_units_of

import torch
from time import time
import numpy as np
import mrcfile
try:
    import openmm.unit as unit
except ImportError: 
    import simtk.unit as unit 

from torch import Tensor
from pytorch3d import transforms
import matplotlib.pyplot as plt


def plot2mat(f_out, X, Y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.matshow(X)
    ax2.matshow(Y)
    fig.savefig(f_out, bbox_inches="tight")


def make_grid(L: int, apix: float, origin: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    L_max = (L-1)/2. * apix
    L_min = -L_max
    grid_x = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[0]
    grid_y = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[1]
    grid_z = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[2]
    xs, ys, zs = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_3d = torch.stack([xs, ys, zs], dim=-1)
    print(grid_3d.shape)
    return grid_3d, L_min, L_max


def make_projection(atoms, grid_3d, res) -> np.ndarray:
    dtype = grid_3d.dtype
    t0 = time()
    vol = atoms2map(
        atoms.coords, atoms.radii, atoms.mass, grid_3d, torch.tensor(res, dtype=dtype))
    t1 = time()
    t_vol_gen = t1 - t0
    
    print(f"time cost (create 3D map): {t_vol_gen*1000:.2f} ms")
    
    # proj_img = transform.project_volumes(vol)
    proj_img = vol.sum(-1)
    t2 = time()
    t_cost_proj = t2 - t1
    print(f"time cost (projection step): {t_cost_proj*1000:.2f} ms")
    return proj_img, vol
 

def rotate_coords(coords: Tensor, rotmat: Tensor) -> Tensor:
    """
    Rotate coordinates
    """
    return coords @ torch.t(rotmat)


def deg2rad(x):
    return x*(torch.pi/180.)


def test():
    L = 5
    raw_res = 2.0
    res_scaling = 0.2
    res = raw_res * res_scaling
    apix = 1.
    dtype_all = torch.float32
    dir_data = "/data/datasets/EM/sim/cas9"
    f_ref_pdb = os.path.join(dir_data, "pdb/cas9.pdb")
    f_out_mrcs = os.path.join("tmp", "test_3.mrcs")
    f_dcd = os.path.join(dir_data, "dcd/cas9-strid100.dcd")
    mol_traj = mdtraj.load(f_dcd, top=f_ref_pdb)
    mol_ref = mdtraj.load(f_ref_pdb, top=f_ref_pdb)
    # mol_aligned = mol_traj.superpose(mol_ref)

    traj_xyz = torch.tensor(
            in_units_of(
                mol_traj.xyz,
                units_in="nanometers",
                units_out="angstroms",
                inplace=True),
            dtype=dtype_all,
        )

    ref_xyz = torch.tensor(
        in_units_of(
            mol_ref.xyz,
            units_in="nanometers",
            units_out="angstroms",
            inplace=True),
        dtype=dtype_all,
        )
    

    print(ref_xyz.shape)
    center_coords = ref_xyz.mean(dim=1).squeeze(dim=0)
    print(f"center_coords = {center_coords}")
    print(mol_traj.n_atoms, "atoms")
    N_atoms = mol_traj.n_atoms

   
    center_coords = torch.zeros(3, dtype=dtype_all)   # tmp #

    grid_3d, L_min, _ = make_grid(L, apix, center_coords, dtype=dtype_all)

    element_names = ["C" for i in range(N_atoms)]
    radii = torch.tensor([const.atom_radius(elem) for elem in element_names], dtype=dtype_all)
    mass = torch.tensor(np.array([12.0 for i in range(N_atoms)]), dtype=dtype_all)
    
    atoms = Atoms(
            coords=torch.zeros(N_atoms, 1),
            radii=radii,
            mass=mass
        )

    num_frames = 1
    mrcs_shape = (num_frames, L, L)
    mrcs_out = mrcfile.new_mmap(
        f_out_mrcs, shape=mrcs_shape, mrc_mode=2, overwrite=True)
    mrcs_out.voxel_size = apix
    mrcs_out.header.origin.x = L_min
    mrcs_out.header.origin.y = L_min

    points = torch.tensor(
        [   
            [1., 0., 0.],
            [2., 0., 0.],
            [0., 1., 0.],
            [0., 2., 0.],
            [0., 2., 1.],
            [1., 2., 0.5],
        ])
    N_points = points.shape[0]
    angles = torch.tensor(
        [
            deg2rad(9.),
            deg2rad(11.),
            deg2rad(-16.) 
        ]
    )
    convention = "ZYZ"
    # print(f"L_min = {L_min}")
    # print(f"grid_3d shape: {grid_3d.shape}")
    # print(f"grid_3d x =\n{grid_3d[..., 0]}")
    # print(f"grid_3d y =\n{grid_3d[..., 1]}")
    # print(f"grid_3d z =\n{grid_3d[..., 2]}")
    R = transforms.euler_angles_to_matrix(angles, convention=convention)
    coords = points @ R.transpose(0, 1)
    print(f"sample coords = {coords}")
    atoms = Atoms(
        coords=coords,
        radii=torch.tensor([1.]).repeat(N_points),
        mass=torch.tensor([1.]).repeat(N_points),
    )

    projection, vol = make_projection(atoms, grid_3d, res)
    # print(f"vol = {vol.numpy().round(2)}")
    print(f"projection =\n{projection.numpy().round(2)}")


    vol_before_rotation = atoms2map(
        points, atoms.radii, atoms.mass, grid_3d, torch.tensor(res))
    vol_before_rotation = torch.as_tensor(vol_before_rotation, dtype=torch.float32)
    
    # print(f"vol_before_rotation {vol_before_rotation.dtype}")
    # print(f"vol_before_rotation {vol_before_rotation.numpy().round(2)}")
    # R = R.transpose(0, 1)
    # print(f"R {R.dtype}")
    print(f"R {R.numpy().round(2)}")
    # input volume for transform_volumes must be in (z, y, x) coordinate
    rotmat_for_affine_grid = R.transpose(0, 1)
    vol2 = cemo.tk.transform.transform_volumes(
        vol_before_rotation.transpose(0, -1), 
        rotmat_for_affine_grid.unsqueeze(0),
    )[0].transpose(0, -1)

    projection2 = vol2.sum(dim=-1)
    # print(f"vol2 = {vol2.numpy().round(2)}")
    print(f"projection2 =\n{projection2.numpy().round(2)}")

    # print(f"vol diff\n{(vol - vol2).numpy().round(2)}")
    print(f"vol diff max {torch.max(torch.abs(vol - vol2)).numpy().round(2)}")
    print(f"projection diff\n{(projection - projection2).numpy().round(2)}")

    fig = "tmp/test3_cmp.png"
    plot2mat(fig, projection.numpy(), projection2.numpy())
    assert torch.allclose(vol, vol2, atol=0.3, rtol=1e-1)
    assert torch.allclose(projection, projection2, atol=0.3, rtol=1e-1)



    # def aux(i: int):
    #     print(f"frame {i+1}")
    #     rotmat = transforms.random_rotation(dtype=torch.float32)  # (3, 3)
    #     # atoms.coords = traj_xyz[i, :] @ rotmat.transpose(0, 1)  # (N, 3)

    #     rotmat = 
    #     atoms.coords = point_origin
    #     mrcs_out.data[i, ...] = make_projection(atoms, grid_3d, res)
    #     print("----------")


    # _ = [aux(i) for i in range(num_frames)]

    mrcs_out.flush()
    mrcs_out.close()

    f_out_mrc = "tmp/t3.mrc"
    with mrcfile.new(f_out_mrc, overwrite=True) as out_mrc:
        out_mrc.set_data(vol.numpy().T.astype(np.float32))
        out_mrc.voxel_size = apix
        # out_mrc.header.nxstart = L_min
        # out_mrc.header.nystart = L_min
        # out_mrc.header.nzstart = L_min
        out_mrc.header.origin.x = L_min
        out_mrc.header.origin.y = L_min
        out_mrc.header.origin.z = L_min