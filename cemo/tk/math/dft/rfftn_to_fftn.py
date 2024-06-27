import torch
from typing import List
import cemo.tk.index as index
from cemo.tk.math.dft.enforce_edge_coeff import enforce_edge_coeff
from cemo.tk.math.complex.conj2 import conj2
from cemo.tk import tensor
from cemo.tk.collection import argmin
Tensor = torch.Tensor


def rfftn_to_fftn(
        x: Tensor,
        dims: List[int],
        symm: bool,
        inplace: bool = False,
        enforce: bool = False,
        use_conj2: bool = False,
        ) -> Tensor:
    """
    Convert a half-space Fourier space representation to a full-space one.
    Note: assume the last dimension in dims is the one to be expanded.

    Args:
        x: results returned by torch.fft.rfftn
        dims: a tuple of two integers, indicating the dimensions of 
            the n-dimensional Fourier transform.
        symm: whether the input coefficient matrix x's layout is symmetric
        inplace: if True, x must be of square shape in the dimensions 
            to be transformed
        enforce: whether to enforce the Fourier coefficient symmetry for
            the 0th and N//2-th columns
        use_conj2: whether to use conj2 instead of torch.conj to calculate
            the complex conjudate. This is useful when the input tensor
            has dtype=torch.bfloat16.
            Note: input x's last dimension must have size = 2 (real & imag).

    Returns:
        A tensor of shape (..., N1, N2).
    """
    device = x.device
    old_sizes = [x.size(d) for d in dims]
    N_max = max(old_sizes)
    i_min = argmin(old_sizes)
    dim_min = dims[i_min]

    if inplace:
        x_full = x
    else:
        # note: since we are copying all of x's data buffer and
        # metadata (including gradients) into x_full,
        # x_full should have "requires_grad=False",
        # otherwise x_full will become a leaf node
        # in the computation graph, instead of a copy of x.
        x_full = tensor.pad(
            x, dims=[dim_min], new_sizes=[N_max],
            requires_grad=False)

    # fill the blanks using symmetry
    dim_incomplete = dim_min  # the dimension that is not fully filled
    N = x_full.size(dim_incomplete)
    i_mid = N // 2  # same value regardless N is even or odd

    # Same for both even and odd N
    # Same for both symmetric and asymmetric layout
    # Define indices for the entries to be filled by symmetry
    # note: cannot use slice obj because neg_idx() needs to
    # operate on tensors.
    idxs1 = [torch.arange(0, N, device=device) for _ in dims[:-1]]
    idxs2 = [torch.arange(i_mid+1, N, device=device)]
    idxs = idxs1 + idxs2
    idx = index.make_idx(
        ndim=x_full.ndim, dims=dims,
        idx=index.ix(idxs))
    idx_neg = index.neg_idx(idx, dims=dims)

    if use_conj2:
        fn_conj = conj2
    else:
        fn_conj = torch.conj
    x_full[idx] = fn_conj(x_full[idx_neg])

    # enforce symmetry for 0-th and N//2-th in dimension dims[-1]
    if enforce:
        x_full = enforce_edge_coeff(
            x_full, dims=dims, symm=symm, use_conj2=use_conj2)

    return x_full
