from typing import Union
DoL = Union[dict, list]


def dict_get_item(d: DoL, i: int) -> DoL:
    """
    Get the i-th item from the input d of type Dict[Any, list]

    Args:
        d: input dictionary
        i: index

    Returns:
        a dict of lists with the i-th item
    """
    if type(d) is list:
        return d[i]
    elif type(d) is dict:
        return {k: dict_get_item(v, i) for k, v in d.items()}
    else:
        raise ValueError(f"unsupported type: {type(d)}")
