from torch import Tensor
from cemo.tk.math.dft import fftn, ifftn


def add_ctf(img: Tensor, ctf: Tensor) -> Tensor:
    """
    Add CTF to images

    Args:
        img: an image of shape (L, L) or a batch of images (N, L, L)
        ctf: a 2D CTF tensor of shape (L, L) or a batch (N, L, L)
    Returns:
        image or images with CTF added
    """
    img_fourier = fftn(img, dims=(-2, -1), symm=True, use_rfft=False)
    img_with_ctf_fourier = img_fourier * ctf
    img_ctf_real = ifftn(
        img_with_ctf_fourier, dims=(-2, -1), symm=True, use_rfft=False).real
    return img_ctf_real
