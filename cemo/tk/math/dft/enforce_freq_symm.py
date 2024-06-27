import torch
from typing import Iterable, Optional
import cemo.tk.index as index
from cemo.tk.math.dft.nyquist_edge_ids import nyquist_edge_ids
from cemo.tk.math.dft.enforce_symm import enforce_symm
from cemo.tk.math.dft.half_index_edge import half_index_edge
Tensor = torch.Tensor


def enforce_freq_symm(
        x: Tensor,
        dims: Iterable[int],
        sdims: Iterable[int],
        symm: bool,
        inplace: bool = False,
        rfft_dim: Optional[int] = None,
        N: Optional[int] = None,
        debug: bool = False,
        ) -> Tensor:
    """
    Enforce the Fourier frequency symmetry for the N//2-th row/column
    when image size N is even. If N is odd, do nothing and return x.

    Args:
        x: results returned by torch.fft.rfftn
        dims: an iterable of two integers, indicating the dimensions of
            the n-dimensional Fourier transform.
        sdims: symmetry-dimensions along which the symmetry is enforced
        symm: whether the input coefficient matrix x's layout is symmetric
        inplace: whether to modify the input tensor in-place.
        rfft_dim: the dimension along which to perform the rfft.
        N: the image size. If None, N is inferred from
            max([x.shape[i] for i in dims])
        debug: whether to print debug information

    Returns:
        A tensor of shape (..., N1, N2, ...).
    """
    if symm and rfft_dim is not None:
        raise ValueError(
            "Unsupported case: symm=True and rfft_dim is not None")

    # The relation F[k] = F[-k].conj fails when N is odd and symm = True
    # It is much easier to change a symmetric x to an asymmetric x,
    # enforce the symmetry, and change it back again.
    if symm:
        # change to asymmetric layout
        input = torch.fft.ifftshift(x, dim=dims)
    else:
        input = x

    ndim = input.ndim
    dims = index.std_idx(dims, ndim=ndim)
    sdims = index.std_idx(sdims, ndim=ndim)

    if N is None:
        N = max([input.shape[d] for d in dims])

    # If N is odd, there is no need to enforce any symmetry.
    if N % 2 == 1:  # N is odd
        output = input
    else:
        # Note 1: later code for enforcing symmetry only works
        #         for the case where N is even.
        # Note 2: input's layout is always ensured to be asymmetric (see above lines 49-51)
        edge_ids = nyquist_edge_ids(N, symm=False)
        idx_list = half_index_edge(
            input, dims=dims, symm=False, sdims=sdims, edge_ids=edge_ids)

        if debug:
            print(f"N = {N}")
            print("idx_list\n", idx_list)

        def fn_symm(x: Tensor) -> Tensor:
            return -x

        output = enforce_symm(
            input,
            dims=dims,
            sdims=sdims,
            idx_list=idx_list,
            fn_symm=fn_symm,
            inplace=inplace,
            rfft_dim=rfft_dim,
            debug=debug
        )

    if symm:
        output = torch.fft.fftshift(output, dim=dims)

    return output
