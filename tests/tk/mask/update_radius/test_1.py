import numpy as np
from cemo.tk.mask import update_radius


def fn_scheduler(x: int):
    return 1. - x / 10


def test():
    """
    Test the implementation of the update_radius function
    """
    iter_counter = 0
    r_min = 0.1
    r_max = 0.5
    new_radius = update_radius(iter_counter, fn_scheduler, r_min, r_max)
    assert np.isclose(new_radius, 0.5)
    iter_counter = 10
    new_radius = update_radius(iter_counter, fn_scheduler, r_min, r_max)
    assert np.isclose(new_radius, 0.1)
    iter_counter = 5
    new_radius = update_radius(iter_counter, fn_scheduler, r_min, r_max)
    assert np.isclose(new_radius, 0.3)
