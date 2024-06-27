import torch
from torch import Tensor
from typing import Optional


def transform_translation_3d(
        rotmat: Tensor,
        shift: Optional[Tensor] = None,
        scale: float = 1.0) -> Tensor:
    """
    Transform a 2D translation vector into a 3D vector after a 3D rotation.

    Args:
        rotmat: rotation matrix (N, 3, 3)
        shift: translation vector (N, 2). If shift = None,
            then assume no shift.
        scale: scaling factor for the output vector. (default: 1.0)

    Returns:
        a 3D translation vector (N, 3, 1)

    """
    N = rotmat.shape[0]  # batch_size
    dtype = rotmat.dtype
    device = rotmat.device
    if shift is None:
        shift = torch.zeros([N, 3, 1], dtype=dtype, device=device)  # (N, 3, 1)
    else:
        shift_raw = scale * torch.cat(
            [
                shift.to(dtype),
                torch.zeros([N, 1], dtype=dtype, device=device)
            ],
            dim=-1)  # (N, 3)
        shift_reshaped = shift_raw.unsqueeze(dim=1)  # (N, 1, 3)
        shift_rotated = shift_reshaped @ rotmat.transpose(-2, -1)  # (N, 1, 3)
        shift = shift_rotated.transpose(-2, -1)  # (N, 3, 1)

    return shift
