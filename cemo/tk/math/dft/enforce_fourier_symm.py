import torch
from typing import Iterable, Optional, Tuple
from cemo.tk.math.dft.enforce_edge_coeff import enforce_edge_coeff
from cemo.tk.math.dft.enforce_face_coeff import enforce_face_coeff
Tensor = torch.Tensor


def enforce_fourier_symm(
        x: Tensor,
        dims: Iterable[int],
        symm: bool,
        use_conj2: bool = False,
        inplace: bool = True,
        rfft_dim: Optional[int] = None,
        sdims: Optional[Iterable[int]] = None,
        fdims: Optional[Iterable[int]] = None,
        debug: bool = False,
        ) -> Tensor:
    """
    Enforce the Fourier coefficient symmetry for edges and/or faces.

    Args:
        x: results returned by torch.fft.rfftn
        dims: an iterable of two integers, indicating the dimensions of
            the n-dimensional Fourier transform.
        symm: whether the input coefficient matrix x's layout is symmetric
        use_conj2: whether to use conj2 instead of torch.conj to calculate
            the complex conjudate. This is useful when the input tensor
            has dtype=torch.bfloat16.
            Note: input x's last dimension must have size = 2 (real & imag).
        inplace: whether to modify the input tensor in-place.
        rfft_dim: the dimension along which to perform the rfft
        sdims: symmetry-dimensions along which the symmetry is enforced
        fdims: face-dimensions along which the symmetry is enforced
        debug: whether to print debug information

    Returns:
        Same as the input tensor x except that the symmetry is enforced.
    """
    if sdims is not None:
        x = enforce_edge_coeff(
            x,
            dims=dims,
            sdims=sdims,
            symm=symm,
            use_conj2=use_conj2,
            inplace=inplace,
            rfft_dim=rfft_dim,
            debug=debug)

    if fdims is not None:
        x = enforce_face_coeff(
            x,
            dims=dims,
            fdims=fdims,
            symm=symm,
            use_conj2=use_conj2,
            inplace=inplace,
            rfft_dim=rfft_dim,
            debug=debug)
    
    return x
