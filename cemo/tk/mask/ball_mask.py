import torch
from torch import Tensor
from typing import List, Optional
from cemo.tk.mask.dist_mask import dist_mask


def ball_mask(
        size: List[int],
        r: float,
        center: Optional[List[float]] = None,
        inclusive: bool = True,
        ignore_center: bool = False,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
        ) -> Tensor:
    """
    Make a 3D ball mask.

    Args:
        size: the size of the mask, i.e., [H, W, D]
        r: radius of the square mask (float)
        center: center of the mask (x, y)
        inclusive: whether to include the points on the boundary 
            (default: True)
        ignore_center: whether to ignore the center point
        dtype: data type of the mask
        device: device to put the mask on

    Returns:
        a boolean Tensor of size (H, W, D)
        True if the point is inside the ball of radius r.
    """
    assert len(size) == 3
    return dist_mask(
        size=size,
        r=r,
        ord=2,
        center=center,
        inclusive=inclusive,
        ignore_center=ignore_center,
        dtype=dtype,
        device=device,
    )
