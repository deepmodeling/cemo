import os
import argparse
from cemo.parsers import add_all_parsers
import numpy
from pytest import approx


def test():
    f_pkl = os.path.join("..", "data", "poses.pkl")
    f_data_out = os.path.join("tmp", "rotmat_error.txt")
    f_rotmat = os.path.join("..", "data", "align_rotmat.txt")
    f_fig = os.path.join("tmp", "test1.png")
    f_data_expect = os.path.join("expect", "rotmat_error.txt")
    parser = argparse.ArgumentParser()

    _ = add_all_parsers(parser.add_subparsers(dest="subcmd"))
    args = parser.parse_args(
        [
            "diff-rotmat",
            "-i", f_pkl,
            "--align-rotmat", f_rotmat,
            "--fig", f_fig,
            "--fig-title", "test1",
            "--num-bins", "10",
            "-o", f_data_out,
        ]
    )
    args.func(args)
    data_out = numpy.loadtxt(f_data_out)
    data_expect = numpy.loadtxt(f_data_expect)
    assert data_out == approx(data_expect)
