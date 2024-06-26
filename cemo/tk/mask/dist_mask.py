import torch
from torch import Tensor
from typing import List, Optional, Union


def dist_mask(
        size: List[int],
        r: float,
        ord: Union[int, float],
        center: Optional[List[float]] = None,
        inclusive: bool = True,
        dtype: Optional[torch.dtype] = torch.float32,
        device: Optional[torch.device] = None,
        ignore_center: bool = False,
        ) -> Tensor:
    """
    Make an n-dimensional mask based on distance from the center.

    Args:
        size: the size of the mask, e.g., [H, W] or [H, W, D]
        r: radius of the square mask (float)
        ord: order of the norm when calculating distances,
            e.g., 2, float("inf"), float("-inf"), 0, or other in or float.
            see:
            https://pytorch.org/docs/stable/generated/torch.linalg.vector_norm.html
        center: center of the mask (x, y)
        inclusive: whether to include the points on the boundary
            (default: True)
        dtype: data type of the mask
        device: device to put the mask on
        ignore_center: whether to ignore the center point
    Returns:
        a boolean Tensor of the same size where data points
            within the radius is True and False otherwise.
    """
    assert len(size) > 0
    size_tensor = torch.tensor(size, dtype=torch.int64, device=device)

    def make_axis(n: int) -> Tensor:
        return torch.arange(n, dtype=dtype, device=device)

    axes = list(map(make_axis, size_tensor))
    grid = torch.stack(
        torch.meshgrid(axes, indexing='ij'),
        dim=-1)

    # convert center to tensor
    if center is None:
        # do integer division using truncation
        divisor = torch.tensor(2, dtype=torch.int64, device=device)
        center = torch.div(
            size_tensor,
            divisor,
            rounding_mode='trunc'
            ).to(dtype=dtype, device=device)
    else:
        center = torch.tensor(center, dtype=dtype, device=device)

    delta = grid - center
    dist = torch.linalg.vector_norm(delta, ord=ord, dim=-1)

    if inclusive:
        mask = (dist <= r)
    else:
        mask = (dist < r)

    if ignore_center:
        center_idx = tuple([x//2 for x in size])
        mask[center_idx] = False

    return mask
