import torch
from typing import Iterable, List, Optional, Tuple, Callable
import cemo.tk.index as index
Tensor = torch.Tensor
T2 = Tuple[int, int]


def enforce_symm(
        x: Tensor,
        dims: Iterable[int],
        sdims: Iterable[int],
        idx_list: Iterable[T2],
        fn_symm: Callable[[Tensor], Tensor],
        inplace: bool = True,
        rfft_dim: Optional[int] = None,
        debug: bool = False,
        ) -> Tensor:
    """
    Enforce the symmetry for given indices.
    Note: input x must have an asymmetric layout.

    Args:
        x: results returned by torch.fft.rfftn
        dims: an iterable of two integers, indicating the dimensions of
            the n-dimensional Fourier transform.
        sdims: symmetry-dimensions along which the symmetry is enforced
        symm: whether the input coefficient matrix x's layout is symmetric
        idx_list: a list of indices
        fn_symm: a function that enforces the symmetry
        use_conj2: whether to use conj2 instead of torch.conj to calculate
            the complex conjudate. This is useful when the input tensor
            has dtype=torch.bfloat16.
            Note: input x's last dimension must have size = 2 (real & imag).
        inplace: whether to modify the input tensor in-place.
        rfft_dim: the dimension along which to perform the rfft
        N: the image size. If None, N is inferred from
            max([x.shape[i] for i in dims])
        debug: whether to print debug information

    Returns:
        A tensor of shape (..., N1, N2, ...).
    """
    ndim = x.ndim
    dims = index.std_idx(dims, ndim=ndim)
    sdims = index.std_idx(sdims, ndim=ndim)

    if debug:
        print("idx_list\n", idx_list)
        print("x\n", x)

    if rfft_dim is None:
        is_rfft = False
    else:
        is_rfft = True
        rfft_dim = index.std_idx(rfft_dim, ndim=ndim)

    if not inplace:
        x = x.clone()

    def enforce(idx: List[Tensor], sdim: int):
        if isinstance(sdim, Iterable):
            # Note: setting dims=sdim is needed for enforce_face_coeff.
            assert len(sdim) == 2, \
                "len(sdim) must be 2 if sdim is Iterable," \
                f"but got {len(sdim)}."
            idx_neg = index.neg_idx(idx, dims=sdim)
            skip_enforce = (is_rfft and rfft_dim in sdim)
        else:
            # Note:
            # For index.neg_idx's dims option, it should be [sdim],
            # i.e., a list of a single index.
            # This is done to ensure compatibility when is_rfft is True.
            # When is_rfft is True, if we set neg_idx(idx, dims=dims),
            # the results will be wrong because the rfft_dim size
            # is truncated.
            idx_neg = index.neg_idx(idx, dims=[sdim])
            skip_enforce = (is_rfft and rfft_dim == sdim)
            
        if skip_enforce:
            # no need to enforce symmetry for the rfft_dim dimension
            return
        else:
            x[idx] = fn_symm(x[idx_neg])
            if debug:
                print(f"sdim = {sdim}")
                print("idx\n", idx)
                print("idx_neg\n", idx_neg)
                print(f"x[{idx}]\n", x[idx])

    def aux(i: int):
        sdim = sdims[i]
        idx = idx_list[i]
        enforce(idx=idx, sdim=sdim)

    _ = list(map(aux, range(len(idx_list))))

    return x
