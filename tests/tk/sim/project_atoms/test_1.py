from re import I
from cemo.tk.sim import Atoms, GridSpecs, project_atoms
from cemo.tk.sim import atoms2map_numba as atoms2map
from cemo.tk.sim import make_grid_3d
from cemo.io import mrc, pdb
from cemo.tk import const
from cemo.tk import transform
from cemo.tk.sim import make_grid_3d, atoms2proj

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


def plot_mat(f_out, X):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    ax1.matshow(X)
    fig.savefig(f_out, bbox_inches="tight")


def plot_mat2(f_out, X, Y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.matshow(X)
    ax2.matshow(Y)
    fig.savefig(f_out, bbox_inches="tight")

def deg2rad(x):
    return x*(torch.pi/180.)


def test():
    f_out_mrc = "tmp/t1.mrc"
    f_out_mrcs = "tmp/t1.mrcs"
    f_out_proj = "tmp/t1.png"
    num_frames = 1
    dtype = torch.float32


    # make a grid
    L = 5
    grid_shape = (L, L, L)
    res = 1.5
    res_scaling = 0.2
    apix = 1.
    dtype_all = torch.float32
    center_coords = torch.zeros(3, dtype=dtype_all) 

    grid_specs = GridSpecs(
        shape=grid_shape,
        apix=apix,
        center=center_coords,
        res=res,
    )

    # make atoms
    coords = torch.tensor(
        [   
            [1., 0., 0.],
            [2., 0., 0.],
            [0., 1., 0.],
            [0., 2., 0.],
            [0., 2., 1.],
            [1., 2., 0.5],
        ], dtype=dtype)

    coords -= coords.mean(dim=0)
    N_atoms = coords.shape[0]
    radii = torch.tensor([1.]).repeat(N_atoms)
    mass = torch.tensor([1.]).repeat(N_atoms)
    atoms = Atoms(
        coords=coords,
        radii=radii,
        mass=mass,
    )

    euler_angles = torch.tensor(
        [
            deg2rad(0.),
            deg2rad(-0.),
            deg2rad(0.) 
        ]
    )
    convention = "ZYZ"
    R = transforms.euler_angles_to_matrix(euler_angles, convention=convention).transpose(0, 1)

    proj, vol, L_min, _, grid_3d = project_atoms(
        atoms, grid_specs, R, as_numpy=True,
        res_scaling=res_scaling, timeit=True)

    # save the projection mrcs file
    mrcs_shape = (num_frames, L, L)
    mrcs_out = mrcfile.new_mmap(
        f_out_mrcs, shape=mrcs_shape, mrc_mode=2, overwrite=True)
    mrcs_out.voxel_size = apix
    mrcs_out.header.origin.x = L_min[0]
    mrcs_out.header.origin.y = L_min[1]

    print(f"mrcs_out.data.shape = {mrcs_out.data.shape}")
    print(f"proj shape = {proj.shape}")
    mrcs_out.data[0, ...] = proj.astype(np.float32)
    mrcs_out.flush()
    mrcs_out.close()

    # save the volume
    with mrcfile.new(f_out_mrc, overwrite=True) as out_mrc:
        out_mrc.set_data(vol.T.astype(np.float32))
        out_mrc.voxel_size = apix
        out_mrc.header.origin.x = L_min[0]
        out_mrc.header.origin.y = L_min[1]
        out_mrc.header.origin.z = L_min[2]

    # save the projection image as png
    plot_mat(f_out_proj, proj)


    # =========================
    projection = torch.tensor(proj, dtype=dtype)
    vol = torch.tensor(vol, dtype=dtype)
    grid_3d = torch.tensor(grid_3d, dtype=dtype)

    # grid_3d, L_min, _, grid_3d = make_grid(L, apix, center_coords, dtype=dtype)
    vol_before_rotation = atoms2map(
        atoms.coords, atoms.radii, atoms.mass, grid_3d, res, res_scaling)
    vol_before_rotation = torch.as_tensor(vol_before_rotation, dtype=torch.float32)
    
    # print(f"vol_before_rotation {vol_before_rotation.dtype}")
    # print(f"vol_before_rotation {vol_before_rotation.numpy().round(2)}")
    # R = R.transpose(0, 1)
    # print(f"R {R.dtype}")
    print(f"R {R.numpy().round(2)}")
    # input volume for transform_volumes must be in (z, y, x) coordinate
    vol2 = cemo.tk.transform.transform_volumes(
        vol_before_rotation.transpose(0, 2), 
        R.unsqueeze(0),
    )[0]
    projection2 = vol2.sum(dim=0)
    # print(f"vol2 = {vol2.numpy().round(2)}")
    print(f"projection2 =\n{projection2.numpy().round(2)}")

    # print(f"vol diff\n{(vol - vol2).numpy().round(2)}")
    print(f"vol diff max {torch.max(torch.abs(vol - vol2)).numpy().round(2)}")
    print(f"projection diff\n{(projection - projection2).numpy().round(2)}")

    fig = "tmp/t1_cmp.png"
    plot_mat2(fig, projection.numpy(), projection2.numpy())
    assert torch.allclose(vol, vol2, atol=0.3, rtol=1e-1)
    assert torch.allclose(projection, projection2, atol=0.3, rtol=1e-1)
