import numpy as np
from test_tk import (
    cs_file_name,
    check_results,
)
from cemo.io.cs import read


def test_read_with_list_index():
    index = [0, 1, 2]
    cs_fid = 1
    f_cs = cs_file_name(cs_fid)
    result = read(f_cs, index=index)
    expect = np.load(f_cs)[index]
    check_results(result, expect)
