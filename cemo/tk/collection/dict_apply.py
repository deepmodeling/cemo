import torch
from typing import Iterable, Union
Tensor = torch.Tensor
DoT = Union[dict, Tensor]


def dict_apply(d: DoT, fn: callable) -> dict:
    """
    Apply a function to all the values of a dictionary.

    Args:
        d: dictionary or tensor
        fn: function to be applied

    Returns:
        an updated dictionary
    """
    if type(d) is dict:
        return {k: dict_apply(v, fn) for k, v in d.items()}
    else:
        return fn(d)
