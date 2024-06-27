# make a 3D cubic mask
import torch
from torch import Tensor
from typing import List, Optional
from cemo.tk.mask import dist_mask


def cube_mask(
        size: List[int],
        r: float,
        center: Optional[List[float]] = None,
        inclusive: bool = True,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
        ) -> Tensor:
    """
    Make a 3D-cube mask.

    Args:
        size: the size of the mask, i.e., [H, W, D]
        r: radius of the cube mask (float)
        center: center of the mask (x, y)
        inclusive: whether to include the points on the boundary
            (default: True)
        dtype: data type of the mask
        device: device to put the mask on

    Returns:
        a boolean Tensor of size (H, W, D).
        True if the point is inside the cube of edge length 2r.
    """
    assert len(size) == 3
    ord = float("inf")

    return dist_mask(
        size=size,
        r=r,
        ord=ord,
        center=center,
        inclusive=inclusive,
        dtype=dtype,
        device=device,
        )
