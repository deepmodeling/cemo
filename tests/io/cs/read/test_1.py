import numpy as np
from test_tk import (
    cs_file_name,
    check_results,
)
from cemo.io.cs import read


def test_read_with_numpy_index():
    cs_fid = 1
    index = np.array([0, 1, 2])
    f_input = cs_file_name(cs_fid)
    result = read(f_input, index)
    expect = np.load(f_input)[index]
    check_results(result, expect)
