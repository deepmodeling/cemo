import torch
from pytorch3d import transforms
from scipy.spatial.transform import Rotation
import numpy as np
import numpy
from pprint import pprint
import matplotlib.pyplot as plt
import mrcfile
import cemo
from torch import Tensor
import prody
from cemo.tk.sim import Atoms
from cemo.tk.sim.f.atoms2map import atoms2map_numba as atoms2map
NumpyArray = numpy.ndarray


def dtype_all():
    return torch.float32

def plot2mat(f_out, X, Y):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.matshow(X)
    ax2.matshow(Y)
    # print(f"min(X)= {np.amin(X)}")
    # print(f"min(Y)= {np.amin(Y)}")

    # print(f"max(X)= {np.amax(X)}")
    # print(f"max(Y)= {np.amax(Y)}")

    # print(f"mean(X)= {np.mean(X)}")
    # print(f"mean(Y)= {np.mean(Y)}")
    # print(f"std(X)= {np.std(X)}")
    # print(f"std(Y)= {np.std(Y)}")
    fig.savefig(f_out, bbox_inches="tight")


def deg2rad(x):
    return x*(torch.pi/180.)

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


def read_pdb_coords(f: str) -> Tensor:
    x = prody.parsePDB(f)
    return torch.tensor(x.getCoords(), dtype=dtype_all())


def make_grid(L: int, apix: float, origin: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    L_max = (L-1)/2. * apix
    L_min = -L_max
    grid_x = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[0]
    grid_y = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[1]
    grid_z = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[2]
    xs, ys, zs = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_3d = torch.stack([xs, ys, zs], dim=-1)
    return grid_3d, L_min, L_max

def make_grid_zyx(L: int, apix: float, origin: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    L_max = (L-1)/2. * apix
    L_min = -L_max
    grid_x = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[0]
    grid_y = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[1]
    grid_z = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[2]
    xs, ys, zs = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_3d = torch.stack([zs, ys, xs], dim=-1)
    return grid_3d, L_min, L_max

def make_grid_yxz(L: int, apix: float, origin: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    L_max = (L-1)/2. * apix
    L_min = -L_max
    grid_x = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[0]
    grid_y = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[1]
    grid_z = torch.linspace(L_min, L_max, L, dtype=dtype) + origin[2]
    xs, ys, zs = torch.meshgrid(grid_x, grid_y, grid_z, indexing="xy")
    grid_3d = torch.stack([xs, ys, zs], dim=-1)
    return grid_3d, L_min, L_max

def make_grid2(Ls: Tensor, apix: float, origin: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    L_max = (Ls-1)/2. * apix
    L_min = -L_max
    grid_x = torch.linspace(L_min[0], L_max[0], Ls[0], dtype=dtype) + origin[0]
    grid_y = torch.linspace(L_min[1], L_max[1], Ls[1], dtype=dtype) + origin[1]
    grid_z = torch.linspace(L_min[2], L_max[2], Ls[2], dtype=dtype) + origin[2]
    xs, ys, zs = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    grid_3d = torch.stack([xs, ys, zs], dim=-1)
    return grid_3d, L_min, L_max


def main():
    f_cs = "/data/datasets/EM/sim/gballs/g5/g5_cen_sim.cs"
    f_vol_gt = "/data/datasets/EM/sim/gballs/g5/g5_cen_map.mrc"
    f_proj_gt = "/data/datasets/EM/sim/gballs/g5/g5_cen_sim.mrcs"
    f_pdb = "/data/datasets/EM/sim/gballs/g5/g5_cen.pdb"
    f_fig_cmp = "tmp/cross-cmp.png"
    L = 16
    apix = 5.0
    res = 20.0
    cs_rotmats = read_cs_rotmat(f_cs)
    print(cs_rotmats.shape)
    vol_gt = read_mrc(f_vol_gt)
    print(vol_gt.shape)
    vol_projs = rotate_and_project_vol(vol_gt, cs_rotmats)
    print(vol_projs.shape)
    gt_projs = read_mrc(f_proj_gt)
    print(gt_projs.shape)


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

    # Ls = torch.tensor([16, 17, 18])
    # grid2, Ls_min, Ls_max = make_grid2(Ls, apix, center_coords, dtype=dtype_all())
    # print(grid2.shape)
    
    which_proj = 3
    rotmat = cs_rotmats[which_proj]

    # pdb_coords = torch.flip(pdb_coords, dims=[1])
    # print(f"flipped pdb coords: {pdb_coords}")

    # Flip around X-axis
    R_correction1 = torch.tensor(
        [
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
        ]
    )

    # rotate around Z-axis by 90 degrees (counter-clockwise)
    R_correction2 = torch.tensor(
        [
            [0., 1., 0.],
            [-1., 0., 0.],
            [0., 0., 1.],
        ]
    )
    new_coords = (pdb_coords @ rotmat) @ R_correction1 @ R_correction2
    # new_coords = (pdb_coords @ rotmat)
    pdb_vol = atoms2map(
        new_coords,
        radii, mass, grid_3d, torch.tensor(res),
        res_scaling=0.2)

    pdb_proj = pdb_vol.sum(dim=2)

    # plot2mat(
    #     f_fig_cmp,
    #     vol_projs[which_proj].numpy(),
    #     gt_projs[which_proj].numpy())
    plot2mat(
        f_fig_cmp,
        vol_projs[which_proj].numpy(),
        pdb_proj.numpy())

    R_c = R_correction1 @ R_correction2
    print("R_c")
    pprint(R_c)
    print("R_c.T")
    pprint(R_c.T)

    # grid_3d, L_min, _ = make_grid_yxz(L, apix, center_coords, dtype=dtype_all())

main()
