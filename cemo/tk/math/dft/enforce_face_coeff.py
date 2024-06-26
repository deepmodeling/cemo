import torch
from typing import Iterable, List, Optional, Tuple
import cemo.tk.index as index
from cemo.tk.math.complex.conj2 import conj2
from cemo.tk.math import dft
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
        raise ValueError("Unsupported case: symm=True and rfft_dim is not None")

    # The relation F[k] = F[-k].conj fails when N is odd and symm = True
    # It is much easier to change a symmetric x to an asymmetric x,
    # enforce the symmetry, and change it back again.
    if symm:
        x = torch.fft.ifftshift(x, dim=dims)

    ndim = x.ndim
    dims = index.std_idx(dims, ndim=ndim)
    fdims = index.std_idx(fdims, ndim=ndim)
    idx_list = dft.half_index_face(x, dims=dims, fdims=fdims)

    if debug:
        print("idx_list\n", idx_list)
        print("x[..., 0]\n", x[..., 0])

    if rfft_dim is None:
        is_rfft = False
    else:
        is_rfft = True
        rfft_dim = index.std_idx(rfft_dim, ndim=ndim)

    if not inplace:
        x = x.clone()

    def enforce_symm(idx: List[Tensor], fdim: T2):
        assert len(fdim) == 2, "len(fdim) must be 2"
        if use_conj2:
            fn_conj = conj2
        else:
            fn_conj = torch.conj

        if is_rfft and rfft_dim in fdim:
            # no need to enforce symmetry for faces intersecting with the rfft_dim dimension
            return
        else:
            idx_neg = index.neg_idx(idx, dims=fdim)
            x[idx] = fn_conj(x[idx_neg])

    def aux(i: int):
        fdim = fdims[i]
        idx = idx_list[i]
        enforce_symm(idx=idx, fdim=fdim)

    _ = list(map(aux, range(len(idx_list))))

    if symm:
        x = torch.fft.fftshift(x, dim=dims)

    return x
