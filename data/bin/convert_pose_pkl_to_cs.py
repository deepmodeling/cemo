"""
Convert poses from pkl to star file.
YHW 2022.12.6
"""
import pickle
import starfile
import pandas as pd
import numpy
from scipy.spatial.transform import Rotation
import argparse
from pandas import DataFrame
from typing import List, Tuple, Optional
NumpyArray = numpy.ndarray


def deg2rad(x: NumpyArray) -> NumpyArray:
    return x * (numpy.pi / 180.)

def read_pkl(f):
    with open(f, "rb") as IN:
        return pickle.load(IN)

def write_pkl(f, data):
    with open(f, "wb") as OUT:
        pickle.dump(data, OUT)

def read_cdrgn_ctf_pkl(f: str) -> dict:
    ctf_params = read_pkl(f)
    return {
        "image_size": ctf_params[:,0],
        "apix": ctf_params[:,1],
        "defocusU": ctf_params[:,2],
        "defocusV": ctf_params[:,3],
        "defocus_angle_deg": ctf_params[:,4],
        "volt_kV": ctf_params[:,5],
        "cs": ctf_params[:,6],
        "contrast": ctf_params[:,7],
        "phase_shift_deg": ctf_params[:,8],
    }


def cs_dtype() -> List[Tuple[str]]:
    return [
        ('uid', '<u8'),
        ('blob/path', 'S100'),
        ('blob/idx', '<u4'),
        ('blob/shape', '<u4', (2,)),
        ('blob/psize_A', '<f4'),
        ('blob/sign', '<f4'),
        ('blob/import_sig', '<u8'),
        ('ctf/type', 'S1'),
        ('ctf/exp_group_id', '<u4'),
        ('ctf/accel_kv', '<f4'),
        ('ctf/cs_mm', '<f4'),
        ('ctf/amp_contrast', '<f4'),
        ('ctf/df1_A', '<f4'),
        ('ctf/df2_A', '<f4'),
        ('ctf/df_angle_rad', '<f4'),
        ('ctf/phase_shift_rad', '<f4'),
        ('ctf/scale', '<f4'),
        ('ctf/scale_const', '<f4'),
        ('ctf/shift_A', '<f4', (2,)),
        ('ctf/tilt_A', '<f4', (2,)),
        ('ctf/trefoil_A', '<f4', (2,)),
        ('ctf/tetra_A', '<f4', (4,)),
        ('ctf/anisomag', '<f4', (4,)),
        ('ctf/bfactor', '<f4'),
        ('alignments3D/split', '<u4'),
        ('alignments3D/shift', '<f4', (2,)),
        ('alignments3D/pose', '<f4', (3,)),
        ('alignments3D/psize_A', '<f4'),
        ('alignments3D/error', '<f4'),
        ('alignments3D/error_min', '<f4'),
        ('alignments3D/resid_pow', '<f4'),
        ('alignments3D/slice_pow', '<f4'),
        ('alignments3D/image_pow', '<f4'),
        ('alignments3D/cross_cor', '<f4'),
        ('alignments3D/alpha', '<f4'),
        ('alignments3D/alpha_min', '<f4'),
        ('alignments3D/weight', '<f4'),
        ('alignments3D/pose_ess', '<f4'),
        ('alignments3D/shift_ess', '<f4'),
        ('alignments3D/class_posterior', '<f4'),
        ('alignments3D/class', '<u4'),
        ('alignments3D/class_ess', '<f4'),
        ('alignments3D/localpose', '<f4', (3,)),
        ('alignments3D/localshift', '<f4', (2,)),
        ('alignments3D/localfulcrum', '<f4', (3,)),
        ]


def mk_cs_data(
        n: int,
        f_mrcs: str,
        L: int,
        apix: float,
        pose: NumpyArray,
        shift: NumpyArray,
        ctf: Optional[dict]=None
    ) -> NumpyArray:
    """
    Make a cs data file
    """
    d = numpy.recarray(
        (n,), dtype=cs_dtype()
    )
    d["uid"] = 0
    d["blob/path"] = f"{f_mrcs}"
    d["blob/idx"] = numpy.arange(n)
    d["blob/shape"] = numpy.array([L, L])
    d["blob/psize_A"] = apix
    d["blob/sign"] = 1.0
    d["blob/import_sig"] = 0
    d["ctf/type"] = b'imported'
    d["ctf/exp_group_id"] = 0
    if ctf is None:
        d["ctf/accel_kv"] = 300.0
        d["ctf/cs_mm"] = 2.7
        d["ctf/amp_contrast"] = 1.0
        d["ctf/df1_A"] = 0.
        d["ctf/df2_A"] = 0.
        d["ctf/df_angle_rad"] = 0.0
        d["ctf/phase_shift_rad"] = 0.0
        d["ctf/scale"] = 1.0
        d["ctf/scale_const"] = 0.0
        d["ctf/shift_A"] = numpy.zeros(2)
        d["ctf/tilt_A"] = numpy.zeros(2)
        d["ctf/trefoil_A"] = numpy.zeros(2)
        d["ctf/tetra_A"] = numpy.zeros(4)
        d["ctf/anisomag"] = numpy.zeros(4)
        d["ctf/bfactor"] = 0.0
    else:
        print(">>> Import CTF parameters from the input")
        d["ctf/accel_kv"] = ctf["volt_kV"]
        d["ctf/cs_mm"] = ctf["cs"]
        d["ctf/amp_contrast"] = ctf["contrast"]
        d["ctf/df1_A"] = ctf["defocusU"]
        d["ctf/df2_A"] = ctf["defocusV"]
        d["ctf/df_angle_rad"] = deg2rad(ctf["defocus_angle_deg"])
        d["ctf/phase_shift_rad"] = deg2rad(ctf["phase_shift_deg"])
        d["ctf/scale"] = 1.0
        d["ctf/scale_const"] = 0.0
        d["ctf/shift_A"] = numpy.zeros(2)
        d["ctf/tilt_A"] = numpy.zeros(2)
        d["ctf/trefoil_A"] = numpy.zeros(2)
        d["ctf/tetra_A"] = numpy.zeros(4)
        d["ctf/anisomag"] = numpy.zeros(4)
        d["ctf/bfactor"] = 0.0
    d["alignments3D/split"] = 0
    d["alignments3D/shift"] = shift
    d["alignments3D/pose"] = pose
    d["alignments3D/psize_A"] = 0.0
    d["alignments3D/error"] = 0.0
    d["alignments3D/error_min"] = 0.0
    d["alignments3D/resid_pow"] = 0.0
    d["alignments3D/slice_pow"] = 0.0
    d["alignments3D/image_pow"] = 0.0
    d["alignments3D/cross_cor"] = 0.0
    d["alignments3D/alpha"] = 1.0
    d["alignments3D/alpha_min"] = 0.0
    d["alignments3D/weight"] = 0.0
    d["alignments3D/pose_ess"] = 0.0
    d["alignments3D/shift_ess"] = 0.0
    d["alignments3D/class_posterior"] = 0.0
    d["alignments3D/class"] = 0
    d["alignments3D/class_ess"] = 0.0
    d["alignments3D/localpose"] = numpy.zeros(3)
    d["alignments3D/localshift"] = numpy.zeros(2)
    d["alignments3D/localfulcrum"] = numpy.zeros(3)
    return d

def rotmat_to_euler(R: numpy.ndarray, degrees: bool = True):
    return Rotation.from_matrix(R).as_euler("ZYZ", degrees=degrees)


def rotmat_to_rotvec(R: NumpyArray) -> NumpyArray:
    return Rotation.from_matrix(R).as_rotvec()


def save_cs(f_out: str, data: NumpyArray):
    with open(f_out, "wb") as OUT:
        numpy.save(OUT, data)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="input cryoDRGN pkl file")
    p.add_argument("-o", "--output", required=True, help="output cryosparc cs file")
    p.add_argument("--box-size", type=int, required=True, help="box edge length in number of pixels")
    p.add_argument("--apix", type=float, required=True, help="angstroms per pixel")
    p.add_argument("--mrcs", type=str, required=True, help="mrcs file name")
    p.add_argument("--ctf", type=str, default="", required=False, help="ctf file name")
    return p.parse_args()


def main(args: argparse.Namespace):
    L = args.box_size
    f_pose_pkl = args.input
    f_mrcs = args.mrcs
    f_cs = args.output
    apix = args.apix
    R, shifts = read_pkl(f_pose_pkl)
    shifts_pixels = shifts * L
    N = R.shape[0]

    rotvecs = rotmat_to_rotvec(R)
    if args.ctf == "":
        data = mk_cs_data(N, f_mrcs, L, apix, rotvecs, shifts_pixels)
    else:
        ctf_params = read_cdrgn_ctf_pkl(args.ctf)
        data = mk_cs_data(N, f_mrcs, L, apix, rotvecs, shifts_pixels, ctf_params)

    print(data.shape)
    save_cs(f_cs, data)
    print("alignments3D/pose", data["alignments3D/pose"][0])
    print("alignments3D/shift", data["alignments3D/shift"][0])
    print(f_cs)
    
    

if __name__ == "__main__":
    main(parse_args())
