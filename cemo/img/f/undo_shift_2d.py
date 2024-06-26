import torch
from ..t import EM_Image
from cemo.tk.transform import transform_images
import numpy


def undo_shift_2d(
        d: EM_Image,
        interp_mode: str = "bilinear",
        align_corners: bool = True) -> EM_Image:
    """
    Undo the image translation to the input image

    Args:
        d: an EM_Image object

    Returns:
        an updated EM_Image object with random shifts
    """
    images = torch.tensor(d.mrcs.data)  # size (N, H, W)
    N, H, W = images.shape  # number of images, height, width

    cs_shift = numpy.copy(d.cs.data["alignments3D/shift"])
    shift_pixels = torch.tensor(cs_shift, dtype=torch.float32)  # (N, 2)
    shift_ratio = shift_pixels / torch.tensor([H, W]).expand(N, 2)
    inverse_shift_ratio = -1. * shift_ratio

    d.mrcs.data = transform_images(
        images,
        shift_ratio=inverse_shift_ratio,
        interp_mode=interp_mode,
        align_corners=align_corners).numpy()

    d.cs.data["alignments3D/shift"] = numpy.zeros((N, 2))

    return d
