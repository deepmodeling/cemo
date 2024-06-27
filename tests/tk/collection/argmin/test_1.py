import pytest
from cemo.tk.collection import argmin


def test_argmin_normal_case():
    xs = [2, 3, 1, 4, 5]
    assert argmin(xs) == 2


def test_argmin_with_negative_numbers():
    xs = [2, -3, 1, -4, 5]
    assert argmin(xs) == 3


def test_argmin_with_all_same_numbers():
    xs = [5, 5, 5, 5, 5]
    assert argmin(xs) == 0


def test_argmin_with_empty_list():
    with pytest.raises(ValueError):
        argmin([])
