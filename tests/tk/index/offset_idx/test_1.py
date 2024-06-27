import pytest
from cemo.tk.index import offset_idx

@pytest.mark.parametrize(
    (
        "id",
        "offset",
        "neg_only",
        "expect", 
    ),
    [
    (5, 3, False, 8),
    (-5, 3, True, -2),
    (5, 3, True, 5),
    ([1, 2, 3], 2, False, [3, 4, 5]),
    ([-1, -2, -3], 2, True, [1, 0, -1]),
    ([1, 2, 3], 2, True, [1, 2, 3]),
    ((1, 2, 3), 2, False, (3, 4, 5)),
    ((-1, -2, -3), 2, True, (1, 0, -1)),
    ((1, 2, 3), 2, True, (1, 2, 3)),
    ([[1, 2], [3, 4]], 2, False, [[3, 4], [5, 6]]),
    ([[-1, -2], [-3, -4]], 2, True, [[1, 0], [-1, -2]]),
    ([[1, 2], [3, 4]], 2, True, [[1, 2], [3, 4]]),
])
def test_offset_idx(id, offset, neg_only, expect):
    print("="*60)
    print(f"id: {id}")
    print(f"offset: {offset}")
    print(f"neg_only: {neg_only}")
    print("="*60)

    result = offset_idx(id, offset=offset, neg_only=neg_only)
    print(f"result: {result}")
    print(f"expect: {expect}")
    assert offset_idx(id, offset=offset, neg_only=neg_only) == expect


@pytest.mark.parametrize(
    "id, offset", 
    [
        ("unsupported", 2),
        ([1.1], 2),
    ]
)
def test_offset_idx_with_unsupported_type(id, offset):
    with pytest.raises(ValueError):
        offset_idx(id, offset)
