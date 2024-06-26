import argparse
from cemo.parsers import add_all_parsers
import numpy
from pytest import approx


def test():
    f_config = "./config/tmat-1.yml"
    f_rot_err_out = "tmp/run/v2/e2e/5nrl/5nrl_snr_200dB_ctf_50k_voxel-grid/5nrl_snr_200dB_ctf_50k_voxel-grid_rotation_error.txt"
    f_rot_err_expect = "expect/5nrl/5nrl_snr_200dB_ctf_50k_voxel-grid_rotation_error.txt"
    f_shift_err_out = "tmp/run/v2/e2e/5nrl/5nrl_snr_200dB_ctf_50k_voxel-grid/5nrl_snr_200dB_ctf_50k_voxel-grid_shift_error.txt"
    f_shift_err_expect = "expect/5nrl/5nrl_snr_200dB_ctf_50k_voxel-grid_shift_error.txt"

    parser = argparse.ArgumentParser()

    _ = add_all_parsers(parser.add_subparsers(dest="subcmd"))
    args = parser.parse_args(
        [
            "tmat-error",
            "-c", f_config,
            "--skip-volume-align",
            "--rotation-error",
            "--rotation-error-squared",
            "--rotation-error-stat", "median",
            "--shift-error",
            "--shift-error-squared",
            "--shift-error-stat", "mean",
        ]
    )
    args.func(args)
    rot_err_out = numpy.loadtxt(f_rot_err_out)
    rot_err_expect = numpy.loadtxt(f_rot_err_expect)
    shift_err_out = numpy.loadtxt(f_shift_err_out)
    shift_err_expect = numpy.loadtxt(f_shift_err_expect)

    assert rot_err_out == approx(rot_err_expect)
    assert shift_err_out == approx(shift_err_expect)
