def jump(
        iter: int,
        iter_max: int,
        inclusive: bool = True,
        ) -> float:
    """
    A step function with a jump at iter_max.

    Args:
        iter: current iteration
        iter_max: iteration at which the jump occurs
        inclusive: whether to include iter_max in the jump

    Returns:
        If inclusive is True, returns 1. if iter >= iter_max, else 0.
        If inclusive is False, returns 1. if iter > iter_max, else 0.
    """
    if inclusive:
        return float(iter >= iter_max)
    else:
        return float(iter > iter_max)
