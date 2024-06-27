import torch
from typing import Iterable, Optional, Tuple
import cemo.tk.index as index
from cemo.tk.math.complex.conj2 import conj2
from cemo.tk.math.dft.half_index_face import half_index_face
from cemo.tk.math.dft.enforce_symm import enforce_symm
Tensor = torch.Tensor
T2 = Tuple[int, int]


def enforce_face_coeff(
        x: Tensor,
        dims: Iterable[int],
        fdims: Iterable[int],
        symm: bool,
        use_conj2: bool = False,
        inplace: bool = True,
        rfft_dim: Optional[int] = None,
        debug: bool = False,
        ) -> Tensor:
    """
    Enforce the Fourier coefficient symmetry for the 0th and N//2-th columns.

    Args:
        x: results returned by torch.fft.rfftn
        dims: an iterable of two integers, indicating the dimensions of
            the n-dimensional Fourier transform.
        fdims: a list of face-dimensions along which the symmetry is enforced
        symm: whether the input coefficient matrix x's layout is symmetric
        use_conj2: whether to use conj2 instead of torch.conj to calculate
            the complex conjudate. This is useful when the input tensor
            has dtype=torch.bfloat16.
            Note: input x's last dimension must have size = 2 (real & imag).
        inplace: whether to modify the input tensor in-place.
        rfft_dim: the dimension along which to perform the rfft
        fdims: edge face-dimensions along which the symmetry is enforced
        debug: whether to print debug information

    Returns:
        A tensor of shape (..., N1, N2, ...).
    """
    assert len(dims) >= 2, "len(dims) must be >= 2"
    if symm and rfft_dim is not None:
        raise ValueError(
            "Unsupported case: symm=True and rfft_dim is not None")

    # The relation F[k] = F[-k].conj fails when N is odd and symm = True
    # It is much easier to change a symmetric x to an asymmetric x,
    # enforce the symmetry, and change it back again.
    if symm:
        x = torch.fft.ifftshift(x, dim=dims)

    ndim = x.ndim
    dims = index.std_idx(dims, ndim=ndim)
    fdims = index.std_idx(fdims, ndim=ndim)
    idx_list = half_index_face(x, dims=dims, fdims=fdims)

    if debug:
        print("idx_list\n", idx_list)
        print("x[..., 0]\n", x[..., 0])

    if use_conj2:
        fn_symm = conj2
    else:
        fn_symm = torch.conj

    x = enforce_symm(
        x,
        dims=dims,
        sdims=fdims,
        idx_list=idx_list,
        fn_symm=fn_symm,
        inplace=inplace,
        rfft_dim=rfft_dim,
        debug=debug,
        )

    if symm:
        x = torch.fft.fftshift(x, dim=dims)

    return x
