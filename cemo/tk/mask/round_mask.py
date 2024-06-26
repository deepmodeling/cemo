import torch
from torch import Tensor
from typing import List, Optional
from cemo.tk.mask.dist_mask import dist_mask


def round_mask(
        size: List[int],
        r: float,
        center: Optional[List[float]] = None,
        inclusive: bool = True,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
        ignore_center: bool = False,
        ) -> Tensor:
    """
    A mask of 2D round mask

    Args:
        size: the size of the mask, i.e., [H, W]
        r: radius of the square mask (float)
        center: center of the mask (x, y)
        inclusive: whether to include the points on the boundary
            (default: True)
        dtype: data type of the mask
        device: device to put the mask on
        ignore_center: whether to ignore the center point

    Returns:
        a boolean Tensor of size (H, W)
        True if the point is inside the circle.
    """
    assert len(size) == 2
    return dist_mask(
        size=size,
        r=r,
        ord=2,
        center=center,
        inclusive=inclusive,
        dtype=dtype,
        device=device,
        ignore_center=ignore_center,
        )
