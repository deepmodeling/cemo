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

def substars(input_file_name:str, output_file_name:str, mrcs_file_name:str, ind_file_name:str):
    df_in = starfile.read(input_file_name)
    ind = read_pkl(ind_file_name)
    ind=np.array(ind)
    print(ind)
    sub_data = df_in["particles"].iloc[ind].copy(deep=True).reset_index(drop=True)
    
    data = replace_rlnImageName(
        sub_data,
        mrcs_file_name,
    )
    data_output = {
        "optics": df_in["optics"],
        "particles": data,
    }
    starfile.write(data_output, output_file_name, overwrite=True)