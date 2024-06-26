import torch
from typing import Iterable, Optional
from cemo.tk.math.dft.dot_freq_and_shift import dot_freq_and_shift
Tensor = torch.Tensor


def fourier_shift(
        x: Tensor,
        shift: Tensor,
        dims: Iterable[int],
        d: float = 1.0,
        s: Optional[Iterable[int]] = None,
        indexing: str = "ij",
        is_rfft: bool = False,
        symm: bool = False,
        debug: bool = False) -> Tensor:
    """
    Perform translation in Fourier space for the specified dimensions.

    Args:
        x: input Fourier coefficient tensor of shape (B1, ..., Bm, N1, ..., Nn)
            where B1, ..., Bm are batch dimensions and N1, ..., Nn are the
            dimensions to be transformed.
            Note: the order of frequencies for x must follow the same
            convention as torch.fft.fftn,
            i.e., the first dimension is the zero-frequency.
        shift: translation vector of shape (2,) or (B1, ..., Bm, 2)
            unit: in real-space physical units.
            If d=1.0, the unit of shift is pixels.
        dims: dimensions to be transformed of shape (n,)
        d: spacing between adjacent sample points in real space.
            For 2D images, "d" is is the same as pixel size.
        s: signal size in the Fourier-transform dimensions.
            note: if is_rfft=True, then s must be specified explicitly
            due to ambiguities in the output shape of rfft.
            For example, both torch.fft.rfftfreq(4) and
            torch.fft.rfftfreq(5) have a length of 3.
        indexing: order of frequency mesh grid coordinates.
            (default "ij", i.e., (x, y); the alternative is "xy", i.e., (y, x).
            See torch.meshgrid's indexing option.
        is_rfft: whether the frequency tensor should correspond to
            the output of torch.rfft
        symm: whether the input x corresponds to a symmetric frequency-grid
        debug: default False

    Returns:
        modified Fourier coefficient tensor of shape (B1, ..., Bm, N1, .., Nn),
        which represents the Fourier transform of the shifted input tensor.
        F_shifted[k] = exp(-2*pi*j*dot_prod(shift, freq) * F[k],
        where N is the shape vector of the last two dimensions of x.
    """
    if is_rfft and symm:
        raise ValueError("is_rfft and symm cannot both be True")

    device = x.device
    dtype = torch.view_as_real(x).dtype
    shift = shift.to(device=device, dtype=dtype)
    dot_prod = dot_freq_and_shift(
        shift=shift,
        x_shape=x.shape,
        s=s,
        dims=dims,
        d=d,
        indexing=indexing,
        is_rfft=is_rfft,
        symm=symm,
        device=device,
        dtype=dtype,
        debug=debug,
    )  # same shape as x

    pi = torch.pi
    theta = -2 * pi * 1j * dot_prod
    shifted_coeffs = x * torch.exp(theta)
    return shifted_coeffs
