import numpy as np
from test_tk import (
    cs_file_name,
    index_file_name,
    check_results,
    read_pkl,
)
from cemo.io.cs import read


def test_read_with_index_pkl():
    cs_fid = 1
    f_index = index_file_name("pkl")
    f_cs = cs_file_name(cs_fid)
    result = read(f_cs, index=f_index)
    index = read_pkl(f_index)
    expect = np.load(f_cs)[index]
    check_results(result, expect)
