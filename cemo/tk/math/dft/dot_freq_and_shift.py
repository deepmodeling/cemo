import torch
from typing import Optional, Iterable
from cemo.tk.math.dft.freqn import freqn
Tensor = torch.Tensor


def dot_freq_and_shift(
        shift: Tensor,
        x_shape: Iterable[int],
        dims: Iterable[int],
        d: float = 1.0,
        s: Optional[Iterable[int]] = None,
        indexing: str = "ij",
        is_rfft: bool = False,
        symm: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        debug: bool = False) -> Tensor:
    """
    Calculate the dot product of frequency and n-dimensional shift tensor
        for the specified dimensions in Fourier space.

    Args:
        shift: shift vector of shape (nD,) or (B1, ..., Bm, nD)
            where "nD" is the number of dimensions with Fourier-transforms
        x_shape: shape of the input tensor x
            note: input Fourier coefficient tensor has a shape of
            (B1, ..., Bm, N1, ..., Nn)
            where B1, ..., Bm are batch dimensions and N1, ..., Nn are the
            dimensions to be transformed.
            Note: the order of frequencies for x must follow the same
            convention as torch.fft.fftn,
            i.e., the first dimension is the zero-frequency.
        dims: indices for the Fourier-transform dimensions, e.g., [-2, -1]
        d: spacing between adjacent sample points in real space.
            For 2D images, "d" is is the same as pixel size.
        s: signal sizes in the Fourier-transform dimensions, 
            e.g. s=[128, 128] for a 2D Fourier transform.
            note: if is_rfft=True, then s must be specified explicitly
            due to ambiguities in the output shape of rfft.
            For example, both torch.fft.rfftfreq(4) and
            torch.fft.rfftfreq(5) have a length of 3.
        indexing: order of frequency mesh grid coordinates.
            (default "ij", i.e., (x, y); the alternative is "xy", i.e., (y, x).
            See torch.meshgrid's indexing option.
        is_rfft: whether the frequency tensor should correspond to
            the output of torch.rfft
        symm: whether to use symmetric frequency-grid layout.
        debug: default False
        dtype: output dtype
        device: output device

    Returns:
        Dot product of frequency and shift of shape (B1, ..., Bm, N1, .., Nn),
        which represents the Fourier transform of the shifted input tensor.
        Example for the 2D shift (pseudo-code):
            dot = (shift[0]*freq[0] + shift[1]*freq[1])
            return dot

        Then you can use the output "dot" as follows to get
        the shifted Fourier transform.
        F_shifted[k] = exp(-2*pi*j*dot) * F[k],
        where N is the shape vector of the last two dimensions of x.
    """
    x_ndim = len(x_shape)
    shift = shift.to(device=device, dtype=dtype)

    # convert user-defined dims to canonical dims
    # note: muse use list to trigger advanced indexing.
    # otherwise, the result will be a scalar.
    transform_dims = torch.arange(x_ndim)[list(dims)]

    bdim_mask = [
        False if i in transform_dims else True for i in range(x_ndim)]
    x_batch_dims = torch.arange(x_ndim)[bdim_mask]
    x_batch_shape = [x_shape[i] for i in x_batch_dims]

    if is_rfft:
        assert is_rfft and (s is not None), \
            "When is_rfft=True, s cannot be None"
        N_shape = list(s)
    else:
        N_shape = [x_shape[i] for i in transform_dims]

    nD = len(dims)  # number of Fourier-transformed dimensions
    if debug:
        assert shift.size(-1) == nD, \
            f"the last dimension of shift must have {nD} elements, " \
            f"but has {shift.size(-1)} elements"

    if shift.ndim == 1:
        shift_batch_shape = [1] * len(x_batch_shape)
    else:
        shift_batch_shape = list(shift.shape[:-1])
        if debug:
            assert shift_batch_shape == x_batch_shape, \
                f"shift_batch_shape = {shift_batch_shape}, " \
                f"x_batch_shape = {x_batch_shape}"
    # new shape (B1, .., Bm, 1, ..., 1, nD, 1)
    # the [1, ..., 1] part is for broadcasting
    # the [nD, 1] part is for dot product
    shift_new_shape = shift_batch_shape + ([1] * nD) + [shift.size(-1), 1]
    shift_reshaped = shift.view(shift_new_shape)  # (B1, ..., Bm, nD, 1)

    freq = freqn(
        full_shape=N_shape,
        d=d,
        indexing=indexing,
        is_rfft=is_rfft,
        symm=symm,
        dtype=dtype,
        device=device,
    )  # shape (N_1, ..., N_nD, nD)

    if debug:
        assert freq.device == shift_reshaped.device, \
            f"freq.device = {freq.device}, " \
            f"shift_reshaped.device = {shift_reshaped.device}"

    # dot product of frequency and shift
    freq_reshaped = freq.unsqueeze(dim=-2)  # shape (N1, ..., N_nD, 1, nD)
    dot_prod = torch.matmul(
        freq_reshaped,  # shape (N1, ..., N_nD, 1, nD)
        shift_reshaped,  # shape (B1, ..., Bm, 1, ..., 1, nD, 1)
        ).squeeze(dim=-1).squeeze(dim=-1)  # shape (B1, ..., Bm, N1, ..., N_nD)

    if debug:
        if is_rfft:
            expected_dot_prod_shape = torch.Size(
                shift_batch_shape + N_shape[:-1] + [N_shape[-1]//2 + 1]
            )
        else:
            expected_dot_prod_shape = torch.Size(shift_batch_shape + N_shape)

        assert dot_prod.shape == expected_dot_prod_shape, \
            f"indexing={indexing}, is_rfft={is_rfft}, nD={nD}\n" \
            f"x_shape = {x_shape}\n" \
            f"freq_reshaped.shape = {freq_reshaped.shape}\n" \
            f"dot_prod.shape = {dot_prod.shape}\n" \
            f"expected_dot_prod_shape = {expected_dot_prod_shape}"

    return dot_prod
