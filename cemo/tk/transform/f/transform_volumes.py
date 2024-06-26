import torch
import torch.nn.functional as F
from typing import Optional


def transform_volumes(
        volume: torch.Tensor,
        rotmat: torch.Tensor,
        shift:  Optional[torch.Tensor] = None,
        align_corners: bool = False) -> torch.Tensor:
    """
    Transform the volume `volume` with the rotation matrix `rotmat`
    and translation `shift`.

    Args:
        volume: The volume to be transformed. 
            note: input volume voxels MUST corresponds to 
            (n_z, n_y, n_x) coordinates convention, otherwise
            the rotation results will be wrong.
        rotmat: The rotation matrix. (batch_size, 3, 3)
        shift: The translation vector (batch_size, 3, 1)
               in terms of ratio.
               Note: do NOT apply a factor of 2.0 to the shift.
               This factor will be applied within this function.
        align_corners: Whether to align corners in grid_sample

    Returns:
        The transformed volumes (batch_size, n_z, n_y, n_x).
    """
    image_size = volume.shape[0]
    N = rotmat.shape[0]  # batch_size
    dtype = rotmat.dtype
    device = rotmat.device
    if shift is None:
        shift_3d = torch.zeros(
            [N, 3, 1],
            dtype=dtype, device=device)  # (N, 3, 1)
    else:
        # there is a factor of 2 to convert ratio to
        # torch.affine_grid's convension.
        shift_3d = 2.0 * shift
    theta = torch.cat([rotmat, shift_3d], dim=-1)  # (N, 3, 4)
    out_shape = [N, 1, image_size, image_size, image_size]
    grid = F.affine_grid(theta, out_shape, align_corners=align_corners)

    # Note: avoid using repeat() which allocates new memory.
    # expand() is more efficient.
    # (see https://pytorch.org/docs/stable/torch.html#torch.tensor.expand)
    volumes = volume.expand(N, 1, -1, -1, -1)  # (N, 1, n_z, n_y, n_x)
    volumes_transformed = F.grid_sample(
        volumes,
        grid,
        align_corners=align_corners).squeeze(dim=1)  # (N, n_z, n_y, n_x)

    return volumes_transformed
