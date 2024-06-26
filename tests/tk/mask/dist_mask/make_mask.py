import numpy as np
from typing import List, Union


def make_mask(
        shape: List[int],
        r: float,
        ord: Union[int, float],
        center: List[int] = None,
        inclusive: bool = True,
        ignore_center: bool = False,
        ) -> np.ndarray:
    """
    Make a square mask.
    """
    if center is None:
        center = np.array([int(x/2) for x in shape])
    else:
        center = np.array(center)
    axes = [np.arange(x) for x in shape]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    delta = grid - center
    dist = np.linalg.norm(delta, ord=ord, axis=-1)
    if inclusive:
        mask = (dist <= r)
    else:
        mask = (dist < r)
    
    if ignore_center:
        center_idx = tuple([x//2 for x in shape])
        mask[center_idx] = False
    
    return mask
