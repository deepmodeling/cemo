from typing import Callable, Union
from cemo.tk.scheduler.jump import jump
from cemo.tk.scheduler.linear import linear
FnJump = Callable[[int, int, bool], float]
FnLinear = Callable[[int, int], float]
FnScheduler = Union[FnJump, FnLinear]


def make_scheduler(
        name: str,
        iter_max: int,
        inclusive: bool = True,
        ) -> FnScheduler:
    """
    Returns a scheduler function by name.

    Args:
        name: name of the scheduler function
        iter_max: maximum number of iterations
        inclusive: whether to use >= when comparing iter and iter_max
            (default: True)
    Returns:
        scheduler function
    """
    if name == 'jump':
        return lambda iter: jump(iter, iter_max, inclusive=inclusive)
    elif name == 'linear':
        return lambda iter: linear(iter, iter_max)
    else:
        raise ValueError(f'Unknown scheduler: {name}')
