import pytest
from cemo.io.cs.get_shift import get_shift


def test_get_shift_with_invalid_input():
    is_abinit = False
    return_ratio = True
    with pytest.raises(TypeError):
        get_shift(None, return_ratio=return_ratio, is_abinit=is_abinit)
