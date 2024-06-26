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
    f_in_mrcs = os.path.join("..", "data", "one_snr1000_no-ctf_2_uniform_shift.mrcs")
    f_in_cs = os.path.join("..", "data", "one_snr1000_no-ctf_2_uniform_shift.cs")
    f_out_mrcs = os.path.join("tmp", "one_snr1000_no-ctf_2_uniform_shift_undo.mrcs")
    f_out_cs = os.path.join("tmp", "one_snr1000_no-ctf_2_uniform_shift_undo.cs")
    parser = argparse.ArgumentParser()

    _ = add_all_parsers(parser.add_subparsers(dest="subcmd"))
    args = parser.parse_args(
        [
            "img_undo_shift_2d",
            "--in-mrcs", f_in_mrcs,
            "--in-cs", f_in_cs,
            "--out-mrcs", f_out_mrcs,
            "--out-cs", f_out_cs,
        ]
    )
    print(args.subcmd)
    args.func(args)
    cs_input = numpy.load(f_in_cs)
    cs_out = numpy.load(f_out_cs)
    print("cs_input blob shape: ", cs_input[0]["blob/shape"])
    print("cs_input output shift: ", cs_input[0]["alignments3D/shift"])

    print("cs_out blob shape: ", cs_out[0]["blob/shape"])
    print("cs_out output shift: ", cs_out[0]["alignments3D/shift"])
