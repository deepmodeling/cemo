from typing import Callable
FnScheduler = Callable[[int], float]


def update_radius(
        iter: int,
        fn_scheduler: FnScheduler,
        r_min: float,
        r_max: float,
        ) -> float:
    """
    Update the radius of a mask

    Args:
        iter: iteration counter
        fn_scheduler: a scheduler function
        r_min: minimum radius
        r_max: maximum radius

    Returns:
        a radius
    """
    beta = fn_scheduler(iter)
    new_radius = r_min + beta * (r_max - r_min)
    return new_radius
