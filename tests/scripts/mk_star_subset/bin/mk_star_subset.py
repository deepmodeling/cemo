"""
Select frames from an star file


author: Yuhang Wang
date: 2022/01/21
"""

import argparse
from email.policy import default
import numpy as np
import os
import pickle
import mrcfile
import numpy
import starfile
from typing import List
from pandas import DataFrame


def read_lines(f: str) -> List[str]:
    with open(f, "r") as IN:
        return IN.read().splitlines()


def read_txt(f: str) -> str:
    with open(f, "r") as IN:
        return IN.read() 


def read_index(f: str) -> numpy.ndarray:
    return numpy.loadtxt(f, dtype=numpy.int32)


def save_txt(f_out: str, content: str):
    with open(f_out, "w") as OUT:
        OUT.write(content + "\n")


def replace_rlnImageName(data: DataFrame, f_mrcs: str) -> DataFrame:
    def format_i(i: int) -> str:
        return "{:0>8d}@".format(i)

    def update(i: int):
        id = format_i(i+1)
        data.at[i, "index"] = f"{i+1}"
        data.at[i, "rlnImageName"] = f"{id}@{f_mrcs}"

    n_frames = len(data)
    _ = list(map(update, range(n_frames)))

    return data


def read_pkl(f: str) -> List[int]:
    with open(f, "rb") as IN:
        return pickle.load(IN)


def add_args(parser):
    parser.add_argument("-i", "--input", required=True, help="input star file")
    parser.add_argument("-o", "--output", required=True, help="Output star file")
    parser.add_argument("--index", required=True, help="input index file")
    parser.add_argument("--mrcs-file",
                        help="the file name of the corresponding mrcs file")
    return parser


def main(args):
    df_in = starfile.read(args.input)
    ind = read_pkl(args.index)

    sub_data = df_in["particles"].iloc[ind].copy(deep=True).reset_index()
    data = replace_rlnImageName(
        sub_data,
        args.mrcs_file,
    )
    data_output = {
        "optics": df_in["optics"],
        "particles": data,
    }
    starfile.write(data_output, args.output, overwrite=True)
    print(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_args(parser).parse_args())
