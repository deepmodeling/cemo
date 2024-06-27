import os
import numpy as np
from typing import Tuple
from cemo.tk.sim.f import add_gaussian_noise,atoms2proj,make_ctf_params,make_freq_grid_2d,calc_ctf,add_ctf,make_grid_3d
import torch
from pytorch3d import transforms
import mrcfile
from .savemrc import save_mrc
from .plotmat import plot_mat
import pickle
import mdtraj
from mdtraj.utils import in_units_of
try:
    import openmm.unit as unit
except ImportError: 
    import simtk.unit as unit

def save_pkl(f: str, data: object):
    with open(f, "wb") as OUT:
        pickle.dump(data, OUT)

class traj(object):
    def __init__(self,num_angles,window_width,snr,use_ctf) -> None:
        L = 64
        self.L=L
        self.traj_frames = num_angles
        self.N_per_frame = window_width
        self.snr = snr
        self.use_ctf = use_ctf
        self.do_algin_dcd = True  

        shape = (L, L, L)
        self.img_shape = (L, L)
        self.res = 4.
        self.res_scaling = 0.2
        atom_mass = 12.
        atom_radius = 2.
        edge_length = 192.  # unit: angstrom
        self.apix = edge_length/L
        self.show_time = False

        dtype_all = torch.float32
        
        f_input_pdb = "pdb/7bcq_chA_res177-330_noh_cen.pdb"
        prefix = f"7bcq_angle_n{self.traj_frames}_cen"
        f_dcd = f"traj/angle/{prefix}.dcd"
        

        print(f">>> input: {f_input_pdb}")
        print(f">>> intput: {f_dcd}")
        
        mol_traj = mdtraj.load(f_dcd, top=f_input_pdb)
        mol_ref = mdtraj.load(f_input_pdb)


        if self.do_algin_dcd:
            print(">>> Do trajectory alginment")
            mol_traj_aligned = mol_traj.superpose(mol_ref)
        else:
            print(">>> Skip trajectory alignment")
            mol_traj_aligned = mol_traj

        self.traj_xyz = torch.tensor(
                in_units_of(
                    mol_traj_aligned.xyz, #[1:, ...], # skip the initial reference frame
                    units_in="nanometers",
                    units_out="angstroms",
                    inplace=True),
                dtype=dtype_all,
            )
        self.N_stack_frames = self.traj_xyz.shape[0] * self.N_per_frame
        
        atom_coords = self.traj_xyz[0,...]
        print(f">>> traj shape: {self.traj_xyz.shape}")
        print(f">>> atom coords shape: {atom_coords.shape}")
        print(f">>> N_stack_frames: {self.N_stack_frames}")

        N_atoms = atom_coords.size(dim=0)
        self.radii = torch.tensor([atom_radius]).repeat(N_atoms)
        self.mass = torch.tensor([atom_mass]).repeat(N_atoms)
        print(f">>> max coords: {atom_coords.max().round()}")
        print(f">>> min coords:  {atom_coords.min().round()}")
        center_coords = self.traj_xyz.mean(dim=1).mean(dim=0) 
        print(f">>> atom_coords shape {atom_coords.shape}")
        print(f">>> average mol. center: {center_coords.round().numpy()}")
        print("center of all frames")
        print(self.traj_xyz.mean(dim=1))
        
        self.grid_3d, self.L_min, _ = make_grid_3d(
            shape, self.apix, center_coords, dtype=dtype_all)

        self.all_rotmats=transforms.random_rotations(self.N_stack_frames, dtype=dtype_all)

    def rand_proj(self,i: int, get_vol: bool = False, show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if (i+1) % 1000 == 0 and show_progress: 
            print(f"frame {i+1}")
        R = self.all_rotmats[i]
        i_frame = i // self.N_per_frame
        atom_coords = self.traj_xyz[i_frame, ...]
        coords_rotated = atom_coords @ R
        proj, vol = atoms2proj(
            coords_rotated, self.radii, self.mass, self.grid_3d, self.res, self.self.res_scaling,
            axis_order="zyx",
            timeit=self.show_time)
        
        vol_numpy = vol.numpy().astype(np.float32)

        if self.snr > 0:
            proj = add_gaussian_noise(
                proj, self.snr, debug=False)

        if self.use_ctf:
            ctf_params = make_ctf_params()
            freq2d_grid = make_freq_grid_2d(self.img_shape, self.apix, indexing="xy")
            ctf = calc_ctf(freq2d_grid, *ctf_params)
            proj = add_ctf(proj, ctf)

        proj_numpy = proj.numpy().astype(np.float32)

        if get_vol:
            return (proj_numpy, vol_numpy)
        elif self.use_ctf:
            return (proj_numpy, ctf_params)
        else:
            return proj_numpy

    def gen_all_mrcs(self):
        # save the projection mrcs file and ctf params
        atom_radius = 2.
        snr=self.snr
        L=self.L
        prefix = f"7bcq_angle_n{self.traj_frames}_cen"
        suffix = f"d{L}_w{self.N_per_frame}_r{atom_radius}_res{round(self.res)}_snr{snr}_ctf"
        dir_out = f"sim/{prefix}_d{L}"
        os.makedirs(dir_out, exist_ok=True)
        f_out_mrcs = f"{dir_out}/{prefix}_{suffix}.mrcs"
        f_out_mrc = f"{dir_out}/{prefix}_{suffix}.mrc"
        f_out_proj = f"{dir_out}/{prefix}_{suffix}.png"
        f_pose = f"{dir_out}/{prefix}_{suffix}_pose.pkl"
        f_pose_T = f"{dir_out}/{prefix}_{suffix}_pose_cryodrgn.pkl"
        f_output_ctf = f"{dir_out}/{prefix}_{suffix}_ctf_cryodrgn.pkl"
        if self.use_ctf:
            all_ctf_params = np.zeros((self.N_stack_frames, 9))
            all_ctf_params[:, 0] = L
            all_ctf_params[:, 1] = self.apix
        mrcs_shape = (self.N_stack_frames, L, L)
        mrcs_out = mrcfile.new_mmap(
            self.f_out_mrcs, shape=mrcs_shape, mrc_mode=2, overwrite=True)
        mrcs_out.voxel_size = self.apix
        mrcs_out.header.origin.x = self.L_min[0]
        mrcs_out.header.origin.y = self.L_min[1]
        for i in range(self.N_stack_frames):
            if self.use_ctf:
                mrcs_out.data[i], ctf_params = self.rand_proj(i)
                all_ctf_params[i, 2:] = ctf_params
            else:
                mrcs_out.data[i] = self.rand_proj(i)
        mrcs_out.flush()
        mrcs_out.close()

        # save the volume
        frame_to_save = 0
        atom_coords = self.traj_xyz[frame_to_save, ...]
        proj, vol = atoms2proj(
                atom_coords, self.radii, self.mass, self.grid_3d, self.res, self.res_scaling,
                axis_order="zyx", timeit=self.show_time)
        save_mrc(f_out_mrc, vol.numpy(), self.apix, self.L_min)

        proj_numpy = add_gaussian_noise(
            proj, snr, debug=True).numpy().astype(np.float32)

        # save the projection image as png
        plot_mat(f_out_proj, proj_numpy)

        # save rotation and translation matrices
        shift = np.zeros((self.N_stack_frames, 2))
        all_rotmats_numpy = self.all_rotmats.numpy()
        save_pkl(f_pose, (all_rotmats_numpy, shift))

        all_rotmats_numpy_T = np.swapaxes(all_rotmats_numpy, 1, 2)
        print(f"all_rotmat_T.shape {all_rotmats_numpy_T.shape}")
        save_pkl(f_pose_T, (all_rotmats_numpy_T, shift))

        # save ctf
        if self.use_ctf:
            save_pkl(f_output_ctf, all_ctf_params)
            print("CTF params")
            print(all_ctf_params[:3, :])

        print("outputs:")
        print(f_out_mrcs)
        print(f_out_mrc)
        print(f_out_proj)
        print(f_pose)
        print(f_pose_T)
        
        if self.use_ctf:
            print(f_output_ctf)