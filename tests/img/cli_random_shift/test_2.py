import os
import argparse
from cemo.parsers import add_all_parsers
import numpy
import mrcfile
import matplotlib.pyplot as plt


def plot_frame(f_out: str, data: numpy.array):
    plt.matshow(data)
    plt.savefig(f_out)


def read_mrcs(f: str):
    with mrcfile.open(f) as IN:
        return IN.data


def test():
    f_in_mrcs = os.path.join("..", "data", "one_snr1000_no-ctf_2.mrcs")
    f_in_cs = os.path.join("..", "data", "one_snr1000_no-ctf_2.cs")
    f_out_mrcs = os.path.join("tmp", "one_snr1000_no-ctf_2_gaussian_shift.mrcs")
    f_out_cs = os.path.join("tmp", "one_snr1000_no-ctf_2_gaussian_shift.cs")
    parser = argparse.ArgumentParser()

    _ = add_all_parsers(parser.add_subparsers(dest="subcmd"))
    args = parser.parse_args(
        [
            "img_random_shift_2d",
            "--dist", "gaussian",
            "--in-mrcs", f_in_mrcs,
            "--in-cs", f_in_cs,
            "--out-mrcs", f_out_mrcs,
            "--out-cs", f_out_cs,
            "--x-shift-percent", "0.1",
            "--y-shift-percent", "0.1",
            "--x-std-percent", "0.1",
            "--y-std-percent", "0.1",
        ]
    )
    print(args.subcmd)
    args.func(args)
    cs = numpy.load(f_out_cs)
    print("cs blob shape: ", cs[0]["blob/shape"])
    print("cs output shift: ", cs[0]["alignments3D/shift"])
