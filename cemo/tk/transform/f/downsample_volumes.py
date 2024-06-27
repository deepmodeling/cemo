import torch
from torch import Tensor


def downsamnple_volumes(volumes: Tensor, new_size: int) -> Tensor:
    """
    Downsample volumes to a new size.

    Args:
        volumes (torch.Tensor): Volumes to be downsampled.
        new_size (int): New size of the volumes.

    Returns:
        torch.Tensor: Downsampled volumes.
    """
    assert volumes.shape[-2] == volumes.shape[-1], "Volumes must be square."
    assert volumes.shape[-3] == volumes.shape[-2], "Volumes must be square."
    assert volumes.shape[-3] == 1, "Volumes must be 3D."
    assert volumes.shape[-1] >= new_size, "new_size must be <= original size."
    
    return volumes.squeeze(1)