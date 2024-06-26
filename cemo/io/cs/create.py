"""
Create a cryosparc record array

YHW 2022.11.30
"""
from cemo.io.cs import CryoSparcCS
import numpy
from typing import Tuple


def create(shape: Tuple) -> CryoSparcCS:
    """
    Write a CryoSparcCS object of certain shape

    Args:
        shape: shape of the output 
    """
    dtype = [
        ('uid', '<u8'),
        ('blob/path', 'S30'),
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

    data = numpy.recarray(
        shape=shape,
        dtype=dtype,
    )

    return CryoSparcCS(data=data)
