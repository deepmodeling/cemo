import torch
from typing import Iterable, Optional
Tensor = torch.Tensor


def freqn(
        full_shape: Iterable[int],
        d: float = 1.0,
        indexing: str = "ij",
        is_rfft: bool = False,
        symm: bool = False,
        reverse: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        ) -> Tensor:
    """
    Build a frequency grid for the specified dimensions.

    Args:
        full_shape: number of frequency grid points all dimensions.
            note: specify the full size even if is_rfft is True.
        d: spacing between adjacent sample points in real space.
            For 2D images, "d" is is the same as pixel size.
        indexing: order of frequency mesh grid coordinates.
            (default "ij", i.e., (x, y); the alternative is "xy", i.e., (y, x).
            See torch.meshgrid's indexing option.
        is_rfft: whether the frequency tensor corresponds to
            the output of torch.rfft
        symm: whether the frequency tensor has a symmetric layout.
        reverse: whether to reverse the output along the last dimension.
            e.g., (X, Y, Z) order is changed to (Z, Y, X) order.
        dtype: data type of the frequency tensor.
        device: device of the frequency tensor.

    Returns:
        frequency tensor of shape full_shape+(ndim,) if is_rfft is False,
        otherwise the shape is full_shape[:-1] + (full_shape[-1]//2 + 1, ndim)
    """
    full_shape = tuple(full_shape)  # ensure full_shape is a tuple
    ndim = len(full_shape)
    i_last = ndim - 1
    if ndim == 0:
        return torch.empty(0, dtype=dtype, device=device)

    if indexing == "xy" and ndim >= 2:
        # swap the first two dimensions to ensure the output shape
        # stays the same regardless of the indexing option.
        input_shape = (full_shape[1], full_shape[0]) + full_shape[2:]
    else:
        input_shape = full_shape

    def freq_1d(i: int) -> Tensor:
        n = input_shape[i]
        # note 1: in the case of indexing="xy", freq_list[0] and freq_list[1]
        #         are swapped before doing the index Cartesian product.
        # note 2: rfft only applies to the last dimension
        # note 3: when indexing="xy", for 3D or higher dimensions, since
        #         only the first two dimensions are swapped, only the last
        #         dimension needs to be truncated by rfftfreq.
        if (is_rfft and indexing == "xy" and ndim <= 2 and i == 0) or \
           (is_rfft and indexing == "xy" and ndim > 2 and i == i_last) or \
           (is_rfft and indexing == "ij" and i == i_last):
            # note: rfftfreq only returns positive frequencies,
            # which is different from torch.fft.fftfreq.
            # For consistency with the torch.fft.fftfreq, we return
            # the truncated full-length Fourier frequencies.
            return torch.fft.fftfreq(
                n, d=d, dtype=dtype, device=device)[:n//2+1]
        else:
            return torch.fft.fftfreq(n, d=d, dtype=dtype, device=device)

    freq_list = [freq_1d(i) for i in range(ndim)]
    grid_tuple = torch.meshgrid(freq_list, indexing=indexing)

    if reverse:
        grid_tuple = grid_tuple[::-1]

    asymm_freq = torch.stack(grid_tuple, dim=-1)  # shape: full_shape+(ndim, 2)

    if symm:
        if is_rfft:
            raise ValueError("symm=True is not supported for is_rfft=True")
        # note: the output has ndim+1 dimensions, where the last dimension is
        # the xyz coordinates.
        # fftshift only needs to be applied to all but the last dimension.
        dims = tuple(range(ndim))
        output = torch.fft.fftshift(asymm_freq, dim=dims)
    else:
        output = asymm_freq

    return output
