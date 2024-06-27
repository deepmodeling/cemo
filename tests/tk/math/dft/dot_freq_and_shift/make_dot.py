import torch
from typing import Iterable, Optional, List
from cemo.tk.math.dft.freqn import freqn
Tensor = torch.Tensor


def make_dot(
        shift: Tensor,
        x_shape: Iterable[int],
        s: Iterable[int],
        dims: List[int],
        indexing: str = "ij",
        is_rfft: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        ) -> Tensor:
    """
    Calculate the expected dot product of freq and shift.
    """
    shift = shift.to(device=device, dtype=dtype)
    x_ndim = len(x_shape)
    transform_dims = torch.arange(x_ndim)[list(dims)]

    if is_rfft:
        assert is_rfft and (s is not None), \
            "When is_rfft=True, s cannot be None"
        N_shape = list(s)
    else:
        N_shape = [x_shape[i] for i in transform_dims]

    nD = len(dims)
    bdim_mask = [
        False if i in transform_dims else True for i in range(x_ndim)]
    x_batch_dims = torch.arange(x_ndim)[bdim_mask]
    x_batch_shape = [x_shape[i] for i in x_batch_dims]

    if shift.ndim == 1:
        shift_batch_shape = [1] * len(x_batch_shape)
    else:
        shift_batch_shape = list(shift.shape[:-1])

    shift_new_shape = shift_batch_shape + ([1] * nD) + [shift.size(-1), 1]
    shift_reshaped = shift.view(shift_new_shape)  # (B1, ..., Bm, nD, 1)

    freq = freqn(
        full_shape=N_shape,
        indexing=indexing,
        is_rfft=is_rfft,
        dtype=dtype,
        device=device,
    )  # shape (N_1, ..., N_nD, nD)

    # dot product of frequency and shift
    dot_prod = torch.matmul(
        freq.unsqueeze(dim=-2),  # shape (N1, ..., N_nD, 1, nD)
        shift_reshaped,  # shape (B1, ..., Bm, 1, ..., 1, nD, 1)
        ).squeeze(dim=-1).squeeze(dim=-1)

    return dot_prod

