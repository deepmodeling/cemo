import pytest
from cemo.tk.index.std_idx import std_idx


def test_std_idx_positive():
    ids = [1, 2, 3]
    ndim = 5
    result = std_idx(ids, ndim)
    assert result == [1, 2, 3]


def test_std_idx_negative():
    ids = [-1, -2, -3]
    ndim = 5
    result = std_idx(ids, ndim)
    assert result == [4, 3, 2]


def test_std_idx_mixed():
    ids = [-1, 0, 2]
    ndim = 5
    result = std_idx(ids, ndim)
    assert result == [4, 0, 2]


def test_std_idx_empty():
    ids = []
    ndim = 5
    result = std_idx(ids, ndim)
    assert result == []


def test_std_idx_out_of_range():
    ids = [6, 7, 8]
    ndim = 5
    with pytest.raises(IndexError):
        std_idx(ids, ndim)


def test_std_idx_nested_lists():
    ids = [[-1, 0, 2], [1, -2, -3]]
    ndim = 5
    result = std_idx(ids, ndim)
    assert result == [[4, 0, 2], [1, 3, 2]]


def test_std_idx_nested_tuples():
    ids = [(-1, 0, 2), (1, -2, -3)]
    ndim = 5
    result = std_idx(ids, ndim)
    assert result == [(4, 0, 2), (1, 3, 2)]

