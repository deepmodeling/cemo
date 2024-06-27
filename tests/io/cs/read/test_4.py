import numpy as np
from test_tk import (
    cs_file_name,
    check_results,
)
from cemo.io.cs import read


def test_read_with_none_index():
    cs_fid = 1
    f_cs = cs_file_name(cs_fid)
    index = None
    result = read(f_cs, index=index)
    expect = np.load(f_cs)
    check_results(result, expect)


# def test_read_with_invalid_index():
#     with pytest.raises(ValueError):
#         read('test_file.npy', 123)
