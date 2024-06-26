import torch
from ..t import EM_Image
from cemo.tk.transform import transform_images


def uniform_shift_2d(
        d: EM_Image,
        shift_percent: torch.Tensor,
        interp_mode: str = "bilinear",
        align_corners: bool = True) -> EM_Image:
    """
    Add translation to the input image

    Args:
        d: an EM_Image object
        shift_percent (2, ): max percentage of image allowed to shift 
            along X & Y.

    Returns:
        an updated EM_Image object with random shifts
    """
    images = torch.tensor(d.mrcs.data)  # size (N, H, W)
    N, H, W = images.shape  # number of images, height, width

    # First generate a ratio vector withn [-1, 1)
    random_shift_ratio = (torch.rand(N, 2) - 0.5) * 2.0
    # scale down to the allowed ratio
    # shape: (N, 2)
    shift_ratio = random_shift_ratio * shift_percent.expand(N, 2)

    d.mrcs.data = transform_images(
        images,
        shift_ratio=shift_ratio,
        interp_mode=interp_mode,
        align_corners=align_corners).numpy()

    shift_pixels = shift_ratio * torch.tensor([H, W]).expand(N, 2)
    d.cs.data["alignments3D/shift"] = shift_pixels.numpy()
    return d
