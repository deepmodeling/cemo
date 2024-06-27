"""
Simulate cryo-EM images from a pdb file

Yuhang Wang 2023.12.19
"""
import os
import mdtraj
import pickle
from mdtraj.utils import in_units_of
import torch
import numpy as np
import mrcfile
from pytorch3d import transforms
import matplotlib.pyplot as plt
import argparse
from typing import Tuple
from cemo.tk.sim import make_grid_3d, atoms2proj
from cemo.tk.sim import add_gaussian_noise
from cemo.tk.sim import calc_ctf
from cemo.tk.sim import make_freq_grid_2d, make_ctf_params, add_ctf
NumpyArray = np.ndarray


def save_pkl(f: str, data: object):
    with open(f, "wb") as OUT:
        pickle.dump(data, OUT)


def read_pkl(f: str) -> object:
    with open(f, "rb") as IN:
        return pickle.load(IN)


def plot_mat(f_out, X):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    ax1.matshow(X)
    fig.savefig(f_out, bbox_inches="tight")


def deg2rad(x):
    return x*(torch.pi/180.)


def save_mrc(f: str, vol: NumpyArray, apix: float, L_min: NumpyArray):
    with mrcfile.new(f, overwrite=True) as out_mrc:
        out_mrc.set_data(vol.astype(np.float32))
        out_mrc.voxel_size = apix
        out_mrc.header.origin.x = L_min[0]
        out_mrc.header.origin.y = L_min[1]
        out_mrc.header.origin.z = L_min[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-pdb", help="input pdb file", required=True)
    p.add_argument("--input-dcd", help="input dcd file", required=True)
    p.add_argument("--image-size", type=int, help="image size", required=True)
    p.add_argument("--output-prefix", type=str, required=True,
                   help="output prefix")
    p.add_argument("--window-width", type=int, required=True,
                   help="number projections per trajectory frame")
    p.add_argument("--snr", type=float, default=0., required=True, 
                   help="signal to noise ratio")
    p.add_argument("--use-ctf", action="store_true",
                   help="whether to add CTF effects")
    p.add_argument("--ctf-pool", type=str, default=None,
                   help="ctf pool file (cryodrgn ctf pkl format)")
    return p.parse_args()


def main(args):
    f_input_pdb = args.input_pdb
    f_input_dcd = args.input_dcd
    L = args.image_size
    N_per_frame = args.window_width
    snr = args.snr
    use_ctf = args.use_ctf

    # must do alignment, otherwise the rotation will lead to 
    # unwanted image shift
    do_algin_dcd = True

    shape = (L, L, L)
    img_shape = (L, L)
    res = 4.
    res_scaling = 0.2
    atom_mass = 12.
    atom_radius = 2.
    edge_length = 192.  # unit: angstrom
    apix = edge_length/L
    show_time = False

    dtype_all = torch.float32

    dir_out = os.path.dirname(args.output_prefix)
    os.makedirs(dir_out, exist_ok=True)
    f_out_mrcs = f"{args.output_prefix}.mrcs"
    f_out_mrc = f"{args.output_prefix}.mrc"
    f_out_proj = f"{args.output_prefix}.png"
    f_pose = f"{args.output_prefix}_pose.pkl"
    f_pose_T = f"{args.output_prefix}_pose_cryodrgn.pkl"
    f_output_ctf = f"{args.output_prefix}_ctf_cryodrgn.pkl"

    print(f">>> input: {f_input_pdb}")
    print(f">>> intput: {f_input_dcd}")

    #==================================
    mol_traj = mdtraj.load(f_input_dcd, top=f_input_pdb)
    mol_ref = mdtraj.load(f_input_pdb)

    # ==========================
    # Trajectory alignment is required 
    # to make sure the center of each trajectory frame is [0, 0, 0]
    # ==========================
    if do_algin_dcd:
        print(">>> Do trajectory alginment")
        mol_traj_aligned = mol_traj.superpose(mol_ref)
    else:
        print(">>> Skip trajectory alignment")
        mol_traj_aligned = mol_traj

    traj_xyz = torch.tensor(
            in_units_of(
                mol_traj_aligned.xyz, #[1:, ...], # skip the initial reference frame
                units_in="nanometers",
                units_out="angstroms",
                inplace=True),
            dtype=dtype_all,
        )
    N_stack_frames = traj_xyz.shape[0] * N_per_frame
    
    atom_coords = traj_xyz[0,...]
    print(f">>> traj shape: {traj_xyz.shape}")
    print(f">>> atom coords shape: {atom_coords.shape}")
    print(f">>> N_stack_frames: {N_stack_frames}")

    N_atoms = atom_coords.size(dim=0)
    radii = torch.tensor([atom_radius]).repeat(N_atoms)
    mass = torch.tensor([atom_mass]).repeat(N_atoms)
    print(f">>> max coords: {atom_coords.max().round()}")
    print(f">>> min coords:  {atom_coords.min().round()}")
    center_coords = traj_xyz.mean(dim=1).mean(dim=0) 
    print(f">>> atom_coords shape {atom_coords.shape}")
    print(f">>> average mol. center: {center_coords.round().numpy()}")
    print("center of all frames")
    print(traj_xyz.mean(dim=1))
    
    grid_3d, L_min, _ = make_grid_3d(
        shape, apix, center_coords, dtype=dtype_all)
 
    all_rotmats = transforms.random_rotations(N_stack_frames, dtype=dtype_all)

    if use_ctf:
        ctf_pool = read_pkl(args.ctf_pool)[:, 2:]
        ctf_pool_size = len((ctf_pool))
    else:
        ctf_pool = None
        ctf_pool_size = 0

    def rand_proj(i: int, get_vol: bool = False, show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if (i+1) % 1000 == 0 and show_progress: 
            print(f"frame {i+1}")
        R = all_rotmats[i]
        i_frame = i // N_per_frame
        atom_coords = traj_xyz[i_frame, ...]
        coords_rotated = atom_coords @ R
        proj, vol = atoms2proj(
            coords_rotated, radii, mass, grid_3d, res, res_scaling,
            axis_order="zyx",
            timeit=show_time)
        
        vol_numpy = vol.numpy().astype(np.float32)

        # add CTF first
        if use_ctf:
            i_ctf = np.random.randint(low=0, high=ctf_pool_size)
            ctf_params = ctf_pool[i_ctf]
            freq2d_grid = make_freq_grid_2d(img_shape, apix, indexing="xy")
            ctf = calc_ctf(freq2d_grid, *ctf_params)
            proj = add_ctf(proj, ctf)

        # then add noise
        if snr > 0:
            proj = add_gaussian_noise(proj, snr, debug=False)
        
        proj_numpy = proj.numpy().astype(np.float32)

        if get_vol:
            return (proj_numpy, vol_numpy)
        elif use_ctf:
            return (proj_numpy, ctf_params)
        else:
            return proj_numpy

    # save the projection mrcs file and ctf params
    if use_ctf:
        all_ctf_params = np.zeros((N_stack_frames, 9))
        all_ctf_params[:, 0] = L
        all_ctf_params[:, 1] = apix
    mrcs_shape = (N_stack_frames, L, L)
    mrcs_out = mrcfile.new_mmap(
        f_out_mrcs, shape=mrcs_shape, mrc_mode=2, overwrite=True)
    mrcs_out.voxel_size = apix
    mrcs_out.header.origin.x = L_min[0]
    mrcs_out.header.origin.y = L_min[1]
    for i in range(N_stack_frames):
        if use_ctf:
            mrcs_out.data[i], ctf_params = rand_proj(i)
            all_ctf_params[i, 2:] = ctf_params
        else:
            mrcs_out.data[i] = rand_proj(i)
    mrcs_out.flush()
    mrcs_out.close()

    # save the volume
    frame_to_save = 0
    atom_coords = traj_xyz[frame_to_save, ...]
    proj, vol = atoms2proj(
            atom_coords, radii, mass, grid_3d, res, res_scaling,
            axis_order="zyx", timeit=show_time)
    save_mrc(f_out_mrc, vol.numpy(), apix, L_min)

    # add ctf first
    if use_ctf:
        ctf_params = make_ctf_params()
        freq2d_grid = make_freq_grid_2d(img_shape, apix, indexing="xy")
        ctf = calc_ctf(freq2d_grid, *ctf_params)
        proj = add_ctf(proj, ctf)

    # then add noise
    if snr > 0:
        proj = add_gaussian_noise(proj, snr, debug=True)
    
    proj_numpy = proj.numpy().astype(np.float32)
    

    # save the projection image as png
    plot_mat(f_out_proj, proj_numpy)

    # save rotation and translation matrices
    shift = np.zeros((N_stack_frames, 2))
    all_rotmats_numpy = all_rotmats.numpy()
    save_pkl(f_pose, (all_rotmats_numpy, shift))

    all_rotmats_numpy_T = np.swapaxes(all_rotmats_numpy, 1, 2)
    print(f"all_rotmat_T.shape {all_rotmats_numpy_T.shape}")
    save_pkl(f_pose_T, (all_rotmats_numpy_T, shift))

    # save ctf
    if use_ctf:
        save_pkl(f_output_ctf, all_ctf_params.astype(np.float32))
        print("CTF params")
        print(all_ctf_params[:3, :])

    print("outputs:")
    print(f_out_mrcs)
    print(f_out_mrc)
    print(f_out_proj)
    print(f_pose)
    print(f_pose_T)
    
    if use_ctf:
        print(f_output_ctf)


if __name__ == "__main__":
    main(parse_args())
