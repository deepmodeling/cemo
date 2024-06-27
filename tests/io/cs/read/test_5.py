import pytest
from test_tk import cs_file_name
from cemo.io.cs import read


def test_read_with_invalid_index():
    index = 123
    cs_fid = 1
    f_cs = cs_file_name(cs_fid)
    with pytest.raises(ValueError):
        read(f_cs, index=index)
