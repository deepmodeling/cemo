import torch
from typing import Optional
from cemo.tk.mask.square_mask import square_mask
from cemo.tk.mask.round_mask import round_mask
from cemo.tk.mask.ball_mask import ball_mask
from cemo.tk.filter.butterworth import butterworth
from typing import Union
Tensor = torch.Tensor


def make_mask(
        size: torch.Size,
        r: Union[float, int],
        filter_order: float = 1.0,
        shape: str = "square",
        inclusive: bool = True,
        is_rfft: bool = False,
        symm: bool = True,
        ignore_center: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        ) -> Tensor:
    """
    Make a frequency marching mask

    Args:
        size: size of the output mask
        r: mask radius or Butterworth cutoff freq. ratio
        filter_order: Butterworth filter order
        dtype: data type
        device: device
        shape: mask shape. Choose from [square, round, ball, butterworth]
        inclusive: whether to use <= when testing within the radius r
        is_rfft: whether the mask is for rfft outputs
        symm: whether the mask is symmetric
        ignore_center: whether to ignore the center point

    Returns:
        a binary mask tensor
    """
    if shape == "square":
        fn_mask = square_mask
    elif shape == "round":
        fn_mask = round_mask
    elif shape == "ball":
        fn_mask = ball_mask
    elif shape == "butterworth":
        fn_mask = butterworth
    else:
        raise ValueError(f"Unsupported mask shape: {shape}")

    if symm and is_rfft:
        raise ValueError("Cannot use symmetric mask for rfft")

    if shape in ["square", "round", "ball"]:
        symm_mask = fn_mask(
                size=list(size),
                r=float(r),
                dtype=dtype,
                device=device,
                inclusive=inclusive,
                ignore_center=ignore_center,
        )
    elif shape == "butterworth":
        symm_mask = fn_mask(
                size=list(size),
                cutoff_freq_ratio=float(r),
                order=filter_order,
                high_pass=False,
                squared_butterworth=True,
                symm=True,
                is_rfft=False,
                dtype=dtype,
                device=device,
        )
    else:
        raise NotImplementedError(f"{shape} is not supported")

    if symm:
        mask = symm_mask
    else:
        mask = torch.fft.ifftshift(symm_mask)

    if is_rfft:
        N = mask.shape[-1]
        mid_idx = N // 2
        return mask[..., :(mid_idx + 1)]
    else:
        return mask
