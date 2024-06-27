import torch
from typing import Tuple, List, Union, Optional
from cemo.tk.physics.microscope.calc_ctf import calc_ctf
from cemo.tk.physics.microscope.CTF_Params import CTF_Params
import cemo.tk.math.dft as dft
Tensor = torch.Tensor
TuLi = Union[Tuple, List]


def apply_ctf(
        images: Tensor,
        dims: TuLi,
        ctf_params: CTF_Params,
        freq: Tensor,
        domain: str,
        apix: Optional[float] = None,
        symm: bool = False,
        use_rfft: bool = False,
        enforce_freq_symm: bool = True,
        ) -> Tensor:
    """
    Apply CTF to an input image in real, Fourier, or Hartley domain.

    Args:
        images: input images in real, Fourier, or Hartley domain
        dims: dimensions to be transformed
        ctf_params: CTF parameters
        freq: Fourier-space frequencies of shape:
            (L, L, NDim) or (B, L, L, NDim),
            where NDim is either 2 or 3.
        domain: "real", "fourier", or "hartley"
        apix: pixel size in angstroms (Å)
        is_real_space: whether input images are in real space
        symm: whether to use symmetric layout
        use_rfft: whether to use rfft or fft
        enforce_freq_symm: whether to enforce symmetry for the freq tensor.
            note: this only affects even-sized freq.

    Returns:
        images with CTF applied
    """
    if enforce_freq_symm:
        rfft_dim = 1 if use_rfft else None
        freq = dft.enforce_freq_symm(
            freq,
            dims=(0, 1),
            sdims=(0, 1),
            symm=symm,
            inplace=False,
            rfft_dim=rfft_dim,
            debug=False,
            )

    ctf = calc_ctf(freq, params=ctf_params, apix=apix)

    if domain == "real":
        full_image_sizes = [images.size(dim=i) for i in dims]
        images_ft = ctf * dft.fftn(
            images,
            dims=dims,
            symm_freq=symm,
            symm_real=True,
            use_rfft=use_rfft)
        return dft.ifftn(
            images_ft,
            dims=dims,
            symm_freq=symm,
            symm_real=True,
            s=full_image_sizes,
            use_rfft=use_rfft).real
    elif domain == "fourier" or domain == "hartley":
        return ctf * images
    else:
        raise ValueError("domain must be 'real', 'fourier', or 'hartley'")
