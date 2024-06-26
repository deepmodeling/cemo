from typing import Iterable, Union
ID = Union[Iterable[int], int]


def std_idx(id: ID, ndim: int) -> ID:
    """
    convert indices to standard positive indices.

    Args:
        id: input indices (can be negative)
        ndim: number of total dimensions

    Returns:
        standard indices
    """
    all_ids = list(range(ndim))
    if type(id) is int:
        return all_ids[id]
    elif isinstance(id, Iterable):
        output = [std_idx(i, ndim=ndim) for i in id]
        if isinstance(id, tuple):
            return tuple(output)
        else:
            return output
    else:
        raise ValueError(f"Unsupported type: {type(id)}")
