import torch
from torch import Tensor
from .transform_images import transform_images


def rotate_images(
        image_stack: Tensor,
        angle: float,
        interp_mode: str = "bilinear",
        align_corners=False) -> Tensor:
    """
    Rotate input images by an angle
    Args:
        image_stack: a tensor of shape (N, H, W) or (N, C, H, W)
        angle: a float, angle in radians.
        interp_mode: a string, interpolation mode
        align_corners: whether to align the corners of the image
            (default: True)
    Returns:
        a tensor of shape same as the input, 
        i.e., (N, H, W) or (N, C, H, W).
    """
    dtype = image_stack.dtype
    device = image_stack.device
    N = image_stack.size(dim=0)
    rotation_angle = torch.tensor(angle, dtype=dtype, device=device)

    # Create the rotation matrix 
    # and expand it to a shape (N, 2, 3)
    # without allocating new memory.
    # https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html#torch.Tensor.expand
    rotmat = torch.tensor(
        [[torch.cos(rotation_angle), -torch.sin(rotation_angle)],
         [torch.sin(rotation_angle), torch.cos(rotation_angle)]],
        dtype=dtype, device=device).expand(N, 2, 2)  # (N, 2, 2)

    return transform_images(
        image_stack,
        rotmat=rotmat,
        interp_mode=interp_mode,
        align_corners=align_corners)
