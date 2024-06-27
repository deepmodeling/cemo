from re import I
from cemo.tk.sim.f.atoms2map import atoms2map_numba as atoms2map
from cemo.tk.sim import Atoms
from cemo.io import mrc, pdb
from cemo.tk import const
from cemo.tk import transform
import cemo
import os
import mdtraj
from mdtraj.utils import in_units_of
from scipy.spatial.transform import Rotation
from pprint import pprint
import prody


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
        atoms.coords, atoms.radii, atoms.mass, grid_3d, torch.tensor(res, dtype=dtype),
        res_scaling=0.2)
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


def dtype_all():
    return torch.float32


def read_pdb_coords(f: str) -> Tensor:
    x = prody.parsePDB(f)
    return torch.tensor(x.getCoords(), dtype=dtype_all())

def gen_euler():
    angles = torch.tensor(
        [
            deg2rad(0.),
            deg2rad(0.),
            deg2rad(10.) 
        ]
    )
    return angles

def read_cs_rotmat(f: str) -> Tensor:
    d = np.load(f)
    k = "alignments3D/pose"
    rotvecs = d[k]
    print(rotvecs.shape)
    return torch.tensor(
        Rotation.from_rotvec(rotvec=rotvecs).as_matrix(),
        dtype=dtype_all(),
    )


def read_mrc(f: str) -> Tensor:
    with mrcfile.open(f) as IN:
        return torch.tensor(IN.data, dtype=dtype_all())


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
    test_id = 5
    f_cs = "../data/g5/g5_cen_sim.cs"
    f_vol_gt = "../data/g5/g5_cen_map.mrc"
    f_pdb = "../data/g5/g5_cen.pdb"
    L = 16
    apix = 5.0
    res = 20.0
    cs_rotmats = read_cs_rotmat(f_cs)
    print(cs_rotmats.shape)
    vol_gt = read_mrc(f_vol_gt)
    print(vol_gt.shape)


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
    center_coords = torch.zeros(3, dtype=dtype_all())   # tmp #
    grid_3d, L_min, _ = make_grid(L, apix, center_coords, dtype=dtype_all())

    angles = torch.tensor(
        [
            [
                deg2rad(45.),
                deg2rad(0.),
                deg2rad(0.) 
            ],
        ]
    )
    convention = "ZYZ"

    which_angle = 0

    rotmat = torch.tensor(
        Rotation.from_euler(convention, angles[which_angle], degrees=False).as_matrix(),
        dtype=dtype_all()).transpose(0, 1)
    new_coords = (pdb_coords @ rotmat)

    # new_coords = (pdb_coords @ rotmat)
    pdb_vol = atoms2map(
        new_coords,
        radii, mass, grid_3d, torch.tensor(res),
        res_scaling=0.2)

    pdb_proj = pdb_vol.sum(dim=2)


    R_correction = torch.tensor(
        [
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 0., 1.],
        ]
    )
    rotmat_csparc = rotmat @ R_correction
    vol_projs = rotate_and_project_vol(vol_gt, rotmat_csparc.unsqueeze(0))



    fig = f"tmp/test{test_id}_cmp.png"
    plot2mat(fig, 
        pdb_proj.numpy(),
        vol_projs[0].numpy())