import torch
Tensor = torch.Tensor


def post_process_filter(
        filter: Tensor,
        is_rfft: bool = False,
        symm: bool = True,
        ) -> Tensor:
    """
    Post-process the filter to handle symmetry and rfft correctly.

    Args:
        filter: filter with a symmetric freq. grid layout
        is_rfft: whether the mask is for rfft outputs
        symm: whether the mask is symmetric

    Returns:
        a binary mask tensor
    """
    if symm and is_rfft:
        raise ValueError("Cannot use symmetric mask for rfft")

    if symm:
        output = filter
    else:
        output = torch.fft.ifftshift(filter)

    if is_rfft:
        N = output.shape[-1]
        mid_idx = N // 2
        return output[..., :(mid_idx + 1)]
    else:
        return output
