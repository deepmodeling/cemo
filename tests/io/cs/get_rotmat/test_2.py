import pytest
from cemo.io.cs.get_rotmat import get_rotmat


def test_get_rotmat_with_invalid_input():
    is_abinit = False
    with pytest.raises(TypeError):
        get_rotmat(None, is_abinit=is_abinit)
