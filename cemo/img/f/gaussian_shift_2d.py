import torch
from ..t import EM_Image
from cemo.tk.transform import transform_images


def gaussian_shift_2d(
        d: EM_Image,
        std: torch.Tensor,
        shift_bound: torch.Tensor,
        interp_mode: str = "bilinear",
        align_corners: bool = True) -> EM_Image:
    """
    Add translation to the input image

    Args:
        d: an EM_Image object
        std (2, ): standard deviation of the Gaussian distribution
             in percentage of image width/height
        shift_bound (2, ): max percentage of image allowed to shift 
            along X & Y.

    Returns:
        an updated EM_Image object with random shifts
    """
    images = torch.tensor(d.mrcs.data)  # size (N, H, W)
    N, H, W = images.shape  # number of images, height, width
    # First generate a ratio vector withn [-1, 1)
    
    random_shift_ratio = (torch.randn(N, 2) * std.expand(N, 2))
    # clamp the values outside the bounds
    bound_min = -shift_bound.expand(N, 2)
    bound_max = shift_bound.expand(N, 2)
    shift_ratio = torch.clamp(
            random_shift_ratio,
            min=bound_min,
            max=bound_max,
        )  # shape: (N, 2)

    d.mrcs.data = transform_images(
        images,
        shift_ratio=shift_ratio,
        interp_mode=interp_mode,
        align_corners=align_corners).numpy()

    shift_pixels = shift_ratio * torch.tensor([H, W]).expand(N, 2)
    d.cs.data["alignments3D/shift"] = shift_pixels.numpy()
    return d
