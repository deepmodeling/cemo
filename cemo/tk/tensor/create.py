import torch
from typing import List, Tuple, Optional, Union
TL = Union[Tuple[int, ...], List[int]]
Tensor = torch.Tensor
Number = Union[int, float, Tensor]


def create(
        shape: TL,
        dtype: torch.dtype,
        device: torch.device,
        layout: torch.layout = torch.strided,
        requires_grad: bool = False,
        fill: Optional[Number] = None,
        ) -> torch.Tensor:
    """
    Create a new tensor with the given shape.

    Args:
        shape: the shape of the new tensor
        dtype: the data type of the new tensor
        device: the device of the new tensor
        layout: the layout of the new tensor
        requires_grad: whether the new tensor requires gradient
        fill: the value to be filled in the new tensor (default: None)

    Returns:
        A tensor of shape shape.
    """
    # create a new tensor
    if fill is None:
        x_new = torch.empty(
            shape,
            dtype=dtype,
            device=device,
            layout=layout,
            requires_grad=requires_grad,
            )
    else:
        if type(fill) is torch.Tensor:
            val = fill.to(device=device, dtype=dtype)
        else:
            val = torch.tensor(fill, dtype=dtype, device=device)
        x_new = torch.full(
            shape,
            fill_value=val,
            dtype=dtype,
            device=device,
            layout=layout,
            requires_grad=requires_grad,
            )

    return x_new
