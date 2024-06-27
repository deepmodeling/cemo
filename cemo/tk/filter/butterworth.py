import torch
from typing import Tuple, Optional
from cemo.tk.math import dft
Tensor = torch.Tensor


def butterworth(
        size: Tuple[int],
        cutoff_freq_ratio: float = 0.1,
        order: float = 2.0,
        high_pass: bool = False,
        squared_butterworth: bool = True,
        symm: bool = False,
        is_rfft: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        ) -> Tensor:
    """
    Make an N-dimensional Butterworth filter.

    Args:
        size: size of the output n-dim filter.
        cutoff_freq_ratio: cutoff frequency expressed in terms of
            fraction of the sampling frequency. Note that
            the Nyquist frequency is half the sampling frequency.
            Should be a value between [0, 0.5].
        order: order of the filter which controls the steepness
            of the slope near the cutoff frequency.
            Higher order leads to steeper slopes, i.e., sharper transitions.
        high_pass: whether to return a high-pass filter
        symm: whether to use a symmetric layout in Fourier space
        is_rfft: whether to use rfft or fft
    
    Returns:
        a filter tensor
    """
    assert cutoff_freq_ratio >= 0, f"cutoff_freq_ratio must be >= 0, not {cutoff_freq_ratio}"
    assert cutoff_freq_ratio <= 0.5, f"cutoff_freq_ratio must be <= 0.5, not {cutoff_freq_ratio}"
    q = dft.freqn(
        size,
        d=cutoff_freq_ratio,
        indexing="ij",
        symm=symm,
        is_rfft=is_rfft,
        dtype=dtype,
        device=device,
    )
    q2 = torch.sum(q**2, dim=-1)
    q2_order = torch.pow(q2, order)
    output = 1. / (1. + q2_order)

    if high_pass:
        # (q2_order * output) is analytically the same as 1 - output
        # but numerically more accurate
        output = q2_order * output  

    if not squared_butterworth:
        output = torch.sqrt(output)

    return output
