def argmin(xs: list) -> int:
    """
    Find the index of the minimum value in a list.

    Args:
        xs: a list of numbers

    Returns:
        The index of the minimum value in xs.

    Reference:
    https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
    """
    return min(range(len(xs)), key=xs.__getitem__)
