
def linear(
        iter: int,
        iter_max: int,
        ) -> float:
    """
    A linear function which plateaus at iter_max.

    Args:
        iter: current iteration
        iter_max: iteration at which the plateau occurs

    Returns:
        Returns 1. if iter >= iter_max, else iter/iter_max.
        If iter_max is 0, returns 1.
    """
    if iter_max == 0:
        return 1.0
    else:
        return min(float(iter)/float(iter_max), 1.0)
