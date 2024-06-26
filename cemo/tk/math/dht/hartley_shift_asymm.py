import torch
from typing import Iterable
import cemo.tk.index as index
from cemo.tk.math.dft.dot_freq_and_shift import dot_freq_and_shift
Tensor = torch.Tensor


def hartley_shift_asymm(
        x: Tensor,
        shift: Tensor,
        dims: Iterable[int],
        d: float = 1.0,
        indexing: str = "ij",
        debug: bool = False) -> Tensor:
    """
    Perform translation in Hartley space for the specified dimensions.
    Assume input Hartley coefficient matrix x has an "asymmetric" layout.

    Args:
        x: input Hartley coefficient tensor of shape (B1, ..., Bm, N1, ..., Nn)
            where B1, ..., Bm are batch dimensions and N1, ..., Nn are the
            dimensions to be transformed.
            Note: the order of frequencies for x must follow the same
            convention as torch.fft.fftn,
            i.e., the first dimension is the zero-frequency.
        shift: translation vector of shape (2,) or (B1, ..., Bm, 2).
            unit: in real-space physical units.
            If d=1.0, the unit of shift is pixels.
        dims: dimensions to be transformed of shape (n,)
        d: spacing between adjacent sample points in real space.
            For 2D images, "d" is is the same as pixel size.
        indexing: order of frequency mesh grid coordinates.
            (default "ij", i.e., (x, y); the alternative is "xy", i.e., (y, x).
            See torch.meshgrid's indexing option.
        debug: default False

    Returns:
        modified Hartley coefficient tensor of shape (B1, ..., Bm, N1, .., Nn),
        which represents the Hartley transform of the shifted input tensor.
        H_shifted[k] = cos(theta) * H[k] + sin(theta) * H[-k],
        where theta = 2 * pi * (shift[0] * k[0]/[0] + shift[1] * k[1]/N[1])
        and N is the shape vector of the last two dimensions of x.
    """
    device = x.device
    dtype = x.dtype
    shift = shift.to(device=device, dtype=dtype)
    dot_prod = dot_freq_and_shift(
        shift=shift,
        x_shape=x.shape,
        s=None,
        dims=dims,
        d=d,
        indexing=indexing,
        is_rfft=False,
        symm=False,
        device=device,
        dtype=dtype,
        debug=debug,
    )  # same shape as x

    pi = torch.pi
    theta = 2 * pi * dot_prod  # shape (N1, ..., Nn)
    cos = torch.cos(theta)
    sin = torch.sin(theta)

    H_k = x

    # calculate H[-k]
    i_mesh = torch.meshgrid(
            torch.arange(H_k.size(dims[-2]), device=device),
            torch.arange(H_k.size(dims[-1]), device=device),
            indexing="ij",
        )
    idx = index.make_idx(ndim=H_k.ndim, dims=dims, idx=i_mesh)
    idx_neg = index.neg_idx(idx, dims=dims)
    H_k_reverse = H_k[idx_neg]

    return (cos * H_k) + (sin * H_k_reverse)
