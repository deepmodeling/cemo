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
from typing import Union

FloatOrArray = Union[float, numpy.ndarray]

def load_pkl(f):
    with open(f, "rb") as IN:
        return pickle.load(IN)


def write_pkl(f, data):
    with open(f, "wb") as OUT:
        pickle.dump(data, OUT)

"""
rlnVoltage: 0    300.0
Name: rlnVoltage, dtype: float64
rlnImagePixelSize: 0    4.25
Name: rlnImagePixelSize, dtype: float64
rlnSphericalAberration: 0    2.7
Name: rlnSphericalAberration, dtype: float64
rlnAmplitudeContrast: 0    0.1
Name: rlnAmplitudeContrast, dtype: float64
rlnOpticsGroup: 0    1
Name: rlnOpticsGroup, dtype: int64
rlnImageSize: 0    128
Name: rlnImageSize, dtype: int64
rlnImageDimensionality: 0    2
Name: rlnImageDimensionality, dtype: int64
rlnOpticsGroupName: 0    opticsGroup1
Name: rlnOpticsGroupName, dtype: object
"""


def mk_star_optics(
        volt: float = 300.,
        apix: float = 1.0,
        cs: float = 0.0,
        contrast: float = 1.0,
        image_size: int = 0,
        dim: int = 2,
        group_id: int = 1,
        group_name: str = "opticsGroup1",
        ) -> DataFrame:
    def wrap(x):
        return pd.Series([x])
    optics = {
        "rlnVoltage": wrap(volt),
        "rlnImagePixelSize":  wrap(apix),
        "rlnSphericalAberration":  wrap(cs),
        "rlnAmplitudeContrast":  wrap(contrast),
        "rlnImageSize":  wrap(image_size),
        "rlnImageDimensionality":  wrap(dim),
        "rlnOpticsGroup":  wrap(group_id),
        "rlnOpticsGroupName":  wrap(group_name),
    }
    return pd.DataFrame(optics)


def mk_imaging_params(
        N: int,
        defocusU: FloatOrArray = 0.,
        defocusV: FloatOrArray = 0.,
        defocusAngle: FloatOrArray = 0.,
        phaseShift: FloatOrArray = 0.,
        ctfBfactor: FloatOrArray = 0.,
        opticsGroup: int = 1,
        classNumber: int = 1,
        ) -> DataFrame:

    def rep(x) -> numpy.ndarray:
        return numpy.repeat(x, N)

    def make_values(x: FloatOrArray) -> numpy.ndarray:
        if type(x) is float or type(x) is int:
            return rep(x)
        elif type(x) is numpy.ndarray:
            return x
        else:
            print(">>> WARN: use all zeros as ctf parameters")
            return numpy.zeros(N)
    data = {
        "defocusU": make_values(defocusU),
        "defocusV": make_values(defocusV),
        "defocusAngle": make_values(defocusAngle),
        "phaseShift": make_values(phaseShift),
        "ctfBfactor": make_values(ctfBfactor),
        "opticsGroup": make_values(opticsGroup),
        "classNumber": make_values(classNumber),
    }
    return pd.DataFrame(data)


def mk_rlnImageName(f_mrcs: str, N: int) -> numpy.ndarray:
    def aux(i: int):
        return f"{i+1:09d}@{f_mrcs}"
    return numpy.array([aux(i) for i in range(N)])

"""
rlnImageName: 000001@J216/simulated_particles.mrcs
rlnAngleRot: -90.0
rlnAngleTilt: 90.0
rlnAnglePsi: 90.0
rlnOriginXAngst: 42.5
rlnOriginYAngst: 42.5
rlnDefocusU: 16047.728516
rlnDefocusV: 16172.686523
rlnDefocusAngle: 0.0
rlnPhaseShift: 0.0
rlnCtfBfactor: 0.0
rlnOpticsGroup: 1
rlnClassNumber: 1
"""
def mk_star_particles(
        f_mrcs: str,
        R: numpy.ndarray,
        T: numpy.ndarray,
        L: int,
        apix: float,
        params: pd.DataFrame,
        ) -> DataFrame:
    N = R.shape[0]
    euler = rotmat_to_euler(R, degrees=True)
    shift = T * L * apix
    data = {
        "rlnImageName": mk_rlnImageName(f_mrcs, N),
        "rlnAngleRot": euler[:, 0],
        "rlnAngleTilt": euler[:, 1],
        "rlnAnglePsi": euler[:, 2],
        "rlnOriginXAngst": shift[:, 0],
        "rlnOriginYAngst": shift[:, 1],
        "rlnDefocusU": params["defocusU"],
        "rlnDefocusV": params["defocusV"],
        "rlnDefocusAngle": params["defocusAngle"],
        "rlnPhaseShift": params["phaseShift"],
        "rlnCtfBfactor": params["ctfBfactor"],
        "rlnOpticsGroup": params["opticsGroup"],
        "rlnClassNumber": params["classNumber"],
    }
    return pd.DataFrame(data)


def rotmat_to_euler(R: numpy.ndarray, degrees: bool = True):
    return Rotation.from_matrix(R).as_euler("ZYZ", degrees=degrees)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True, help="input cryoDRGN pkl file")
    p.add_argument("-o", "--output", required=True, help="output star file")
    p.add_argument("--box-size", type=int, required=True, help="box edge length in number of pixels")
    p.add_argument("--apix", type=float, required=True, help="angstroms per pixel")
    p.add_argument("--mrcs", type=str, required=True, help="mrcs file name")
    p.add_argument("--ctf", type=str, default="", required=False, help="ctf file name")
    return p.parse_args()


def main(args: argparse.Namespace):
    L = args.box_size
    f_pose_pkl = args.input
    f_mrcs = args.mrcs
    f_star = args.output
    apix = args.apix
    R, T = load_pkl(f_pose_pkl)
    N = R.shape[0]

    
    if args.ctf != "":
        ctf_params = load_pkl(args.ctf)
        img_params = mk_imaging_params(
            N,
            defocusU=ctf_params[:, 2],
            defocusV=ctf_params[:, 3],
            defocusAngle=ctf_params[:, 4],
            phaseShift=ctf_params[:, 8]
        )
        apix = ctf_params[0, 1]
        volt = ctf_params[0, 5]
        cs = ctf_params[0, 6]
        constrast = ctf_params[0, 7]
        image_size = L
        optics = mk_star_optics(
            volt=volt,
            apix=apix,
            cs=cs,
            contrast=constrast,
            image_size=image_size
            )
    else:
        optics = mk_star_optics(apix=args.apix, image_size=L)
        img_params = mk_imaging_params(N)
    particles = mk_star_particles(f_mrcs, R, T, L, apix, img_params)

    star_data = {"optics": optics, "particles": particles}
    starfile.write(star_data, f_star, overwrite=True)
    print(f_star)

if __name__ == "__main__":
    main(parse_args())
