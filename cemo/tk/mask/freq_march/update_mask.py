
from typing import Optional, Tuple
import logging
import torch
from cemo.tk.mask.freq_march.FreqMarchMask import FreqMarchMask
from cemo.tk.mask import make_mask, update_radius
Tensor = torch.Tensor
logger = logging.getLogger(__name__)


def update_mask(
        iter: int,
        size: Tuple[int, int],
        p: FreqMarchMask,
        dtype: torch.dtype,
        device: torch.device,
        ) -> Tuple[Tensor, float]:
    """
    Update frequency marching mask.

    Args:
        iter: current iteration (type: int)
        p: parameters for frequency marching mask (type: FreqMarchMask)
        dtype: data type for the mask (type: torch.dtype)
        device: device for the mask (type: torch.device)

    Returns:
        a tuple: (mask, new_radius)
        mask: frequency marching mask
        new_radius: updated radius
    """
    new_radius = int(
        update_radius(
            iter=iter,
            fn_scheduler=p.scheduler,
            r_min=p.r_min,
            r_max=p.r_max,
        ))

    new_mask = make_mask(
        size=size,
        r=new_radius,
        shape=p.mask_shape,
        dtype=dtype,
        device=device,
    )

    return (new_mask, new_radius)
