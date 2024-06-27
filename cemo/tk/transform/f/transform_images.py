import torch
from torch.nn import functional as F
from typing import Optional
Tensor = torch.Tensor


def transform_images(
        image_stack: Tensor,
        rotmat: Optional[Tensor] = None,
        shift_ratio: Optional[Tensor] = None,
        interp_mode: str = "bilinear",
        align_corners=False) -> Tensor:
    """
    Add translation to the input image

    Args:
        image_stack: a tensor of shape (N, H, W) or (N, C, H, W)
        rotmat: a 2D rotation matrix (N, 2, 2)
        shift_ratio: a tensor of shape (N, 2)
        align_corners: whether to align the corners of the image
              (default: False)
    Returns:
        a tensor of shape (N, H, W)
    """
    dtype = image_stack.dtype
    device = image_stack.device

    if len(image_stack.size()) == 3:
        # set the number of channels to 1
        # new shape: (N, 1, H, W)
        images = image_stack.unsqueeze(dim=1)
    else:
        images = image_stack

    N = images.size(dim=0)

    if rotmat is None:
        rotmat = torch.eye(
            2, dtype=dtype, device=device).expand(N, -1, -1)  # (N, 2, 2)

    if shift_ratio is None:
        shift_ratio = torch.zeros(N, 2, dtype=dtype, device=device)

    # When using pytorch affine_grid+grid_sample, 
    # a shift of 0.5 actually means a shift of 0.25.
    # Thus the shift should be scaled by a factor of 2.
    shift = shift_ratio * 2.0

    # Create the rotation+translation matrix 
    # and expand it to a shape (N, 2, 3)
    # without allocating new memory.
    # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
    theta = torch.concat([rotmat, shift.reshape(N, 2, 1)], dim=-1)  # (N, 2, 3)

    # output shape: (N, 1, H, W)
    grid = F.affine_grid(
        theta,
        images.size(),
        align_corners=align_corners)

    output = F.grid_sample(
        images,
        grid,
        padding_mode="zeros",
        align_corners=align_corners,
        mode=interp_mode)

    # return a tensor of shape (N, H, W)
    if len(image_stack.size()) == 3:
        return output.squeeze(dim=1)
    else:
        return output
