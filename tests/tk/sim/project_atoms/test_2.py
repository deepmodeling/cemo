from re import I
from cemo.tk.sim import Atoms, GridSpecs, project_atoms
from cemo.tk.sim import atoms2map_numba as atoms2map
from cemo.tk.sim import make_grid_3d
from cemo.io import mrc, pdb
from cemo.tk import const
from cemo.tk import transform
from cemo.tk.sim import make_grid_3d, atoms2proj
import prody
from scipy.spatial.transform import Rotation

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

def dtype_all():
    return torch.float32

def read_pdb_coords(f: str) -> Tensor:
    x = prody.parsePDB(f)
    return torch.tensor(x.getCoords(), dtype=dtype_all())


def rotate_and_project_vol(vol: Tensor, rotmat: Tensor) -> Tensor:
    vol2 = cemo.tk.transform.transform_volumes(
        vol,
        rotmat,
    )
    def proj(i: int):
        return vol2[i].sum(dim=0)

    N = vol2.shape[0]
    return torch.stack(
        list(map(proj, range(N))),
        dim=0)

def test():
    # make a grid
    test_id = 2
    f_pdb = "../data/g5/g5_cen.pdb"
    L = 16
    apix = 5.0
    res = 20.0
    res_scaling = 0.2


    grid_shape = (L, L, L)
    center_coords = torch.zeros(3, dtype=dtype_all()) 

    grid_specs = GridSpecs(
        shape=grid_shape,
        apix=apix,
        center=center_coords,
        res=res,
    )

    # pdb
    pdb_coords = read_pdb_coords(f_pdb)
    pdb_center = pdb_coords.mean(dim=0)
    pdb_coords -= pdb_center
    pdb_center = pdb_coords.mean(dim=0)
    print(f"pdb_center {pdb_center}")
    N_atoms = pdb_coords.shape[0]
    radii = torch.tensor([1.]).repeat(N_atoms)
    mass = torch.tensor([1.]).repeat(N_atoms)
    print(pdb_coords.shape)

    atoms = Atoms(
        coords=pdb_coords,
        radii=radii,
        mass=mass,
    )


    angles = torch.tensor(
        [
            [
                deg2rad(10.),
                deg2rad(12.),
                deg2rad(-15.) 
            ],

            [
                deg2rad(1.),
                deg2rad(-12.),
                deg2rad(15.) 
            ],
        ]
    )
    convention = "ZYZ"

    which_angle = 1

    R = torch.tensor(
        Rotation.from_euler(convention, angles[which_angle], degrees=False).as_matrix(),
        dtype=dtype_all()).transpose(0, 1)

    pdb_proj, vol, L_min, _, grid_3d = project_atoms(
        atoms, grid_specs, R, as_numpy=False,
        res_scaling=res_scaling,
        axis_order="zyx",
        timeit=True)

    print("type of grid_3d")
    print(type(grid_3d))

    vol_gt = atoms2map(
        pdb_coords,
        radii, mass, grid_3d, torch.tensor(res),
        res_scaling=0.2).transpose(0, 2)

    rotmat_csparc = R
    vol_projs = rotate_and_project_vol(vol_gt, rotmat_csparc.unsqueeze(0))

    fig = f"tmp/test{test_id}_cmp.png"
    plot2mat(fig, 
        pdb_proj.numpy(),
        vol_projs[0].numpy())
    print(fig)

