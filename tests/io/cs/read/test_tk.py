import numpy as np
import pickle
import os
from cemo.io.cs import CryoSparcCS


def data_dir() -> str:
    return "../data"


def cs_file_name(which_file: int) -> str:
    if which_file == 1:
        fname = "tg2_n5_with_ctf_and_shift_snr0.1.cs"
    elif which_file == 2:
        fname = "tg2_n10000_with_ctf_and_shift_snr0.1.cs"
    else:
        raise ValueError(f"Unsupported file ID: {which_file}")

    return os.path.join(data_dir(), fname)


def index_file_name(file_type: str) -> str:
    if file_type == "pkl":
        fname = "index_3.pkl"
    elif file_type == "npy":
        fname = "index_3.npy"
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return os.path.join(data_dir(), fname)


def check_results(
        result: CryoSparcCS,
        expect: np.ndarray,
        debug: bool = False):
    assert isinstance(result, CryoSparcCS)
    for k in expect.dtype.names:
        if debug:
            print(k)
            print(f"result: {result.data[k]}")
            print(f"expect: {expect[k]}")
        assert (result.data[k] == expect[k]).all()


def read_pkl(f_input: str):
    with open(f_input, 'rb') as f:
        return np.array(pickle.load(f))
