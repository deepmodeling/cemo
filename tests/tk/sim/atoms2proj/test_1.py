from re import I
from cemo.tk.sim import Atoms
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

def deg2rad(x):
    return x*(torch.pi/180.)


def test():
    L = 5
    res = 2.0
    apix = 1.
    dtype_all = torch.float32
    num_frames = 1
    f_out_mrc = "tmp/t1.mrc"
    f_out_mrcs = "tmp/t1.mrcs"
    f_out_proj = "tmp/t1.png"

    center_coords = torch.zeros(3, dtype=dtype_all)   # tmp #
    grid_3d, L_min, _ = make_grid_3d(
        L, apix, center_coords, dtype=dtype_all)
    

    coords = torch.tensor(
        [   
            [1., 0., 0.],
            [2., 0., 0.],
            [0., 1., 0.],
            [0., 2., 0.],
            [0., 2., 1.],
            [1., 2., 0.5],
        ])
    N_atoms = coords.shape[0]
    radii = torch.tensor([1.]).repeat(N_atoms)
    mass = torch.tensor([1.]).repeat(N_atoms)

    euler_angles = torch.tensor(
        [
            deg2rad(9.),
            deg2rad(11.),
            deg2rad(-16.) 
        ]
    )
    convention = "ZYZ"
    R = transforms.euler_angles_to_matrix(euler_angles, convention=convention)
    
    # rotate the molecule
    coords_rotated = coords @ R.transpose(0, 1)
    
    print(f"Rotated coords = {coords_rotated}")
    
        
    proj, vol = atoms2proj(
        coords_rotated, radii, mass, grid_3d, res, timeit=True)

    # save the projection mrcs file
    mrcs_shape = (num_frames, L, L)
    mrcs_out = mrcfile.new_mmap(
        f_out_mrcs, shape=mrcs_shape, mrc_mode=2, overwrite=True)
    mrcs_out.voxel_size = apix
    mrcs_out.header.origin.x = L_min
    mrcs_out.header.origin.y = L_min   
 
    print(f"mrcs_out.data.shape = {mrcs_out.data.shape}")
    print(f"proj shape = {proj.shape}")
    mrcs_out.data[0, ...] = proj.numpy().astype(np.float32)
    assert torch.allclose(torch.tensor(mrcs_out.data[0]), proj)
    mrcs_out.flush()
    mrcs_out.close()

    # save the volume
    with mrcfile.new(f_out_mrc, overwrite=True) as out_mrc:
        out_mrc.set_data(vol.numpy().T.astype(np.float32))
        out_mrc.voxel_size = apix
        out_mrc.header.origin.x = L_min
        out_mrc.header.origin.y = L_min
        out_mrc.header.origin.z = L_min

    # save the projection image as png
    plot_mat(f_out_proj, proj.numpy())


