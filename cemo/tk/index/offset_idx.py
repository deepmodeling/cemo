from typing import Iterable, Union
LoI = Union[int, Iterable[int]]


def offset_idx(id: LoI, offset: int, neg_only: bool = False) -> Iterable[int]:
    """
    Offset indices by a constant.

    Args:
        id: an integer or an iterable of integers
        offset: the offset
        neg_only: whether to offset only negative indices
    
    Returns:
        Same shape as the input but with the indices offset by the given constant.
    """
    unsupported_types = (str, float)
    if isinstance(id, unsupported_types):
        raise ValueError(f"Unsupported type: {type(id)}")

    if isinstance(id, Iterable):
        output = [offset_idx(i, offset=offset, neg_only=neg_only) for i in id]
        if type(id) is tuple:
            return tuple(output)
        else:
            return output
    else:
        if neg_only:
            return id + offset if id < 0 else id
        else:
            return id + offset
