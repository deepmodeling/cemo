import torch
from typing import Iterable
from cemo.tk.math.dht.hartley_shift_asymm import hartley_shift_asymm
from cemo.tk.math.dht.hartley_shift_symm import hartley_shift_symm
Tensor = torch.Tensor


def hartley_shift(
        x: Tensor,
        shift: Tensor,
        dims: Iterable[int],
        d: float = 1.0,
        indexing: str = "ij",
        symm: bool = False,
        debug: bool = False) -> Tensor:
    """
    Perform translation in Hartley space for the specified dimensions.

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
        symm: whether to use a symmetric freq-grid layout
        debug: default False

    Returns:
        modified Hartley coefficient tensor of shape (B1, ..., Bm, N1, .., Nn),
        which represents the Hartley transform of the shifted input tensor.
        H_shifted[k] = cos(theta) * H[k] + sin(theta) * H[-k],
        where theta = 2 * pi * (shift[0] * k[0]/[0] + shift[1] * k[1]/N[1])
        and N is the shape vector of the last two dimensions of x.
    """
    if symm:
        aux = hartley_shift_symm
    else:
        aux = hartley_shift_asymm

    return aux(
        x,
        shift=shift,
        dims=dims,
        d=d,
        indexing=indexing,
        debug=debug,
    )
