import torch
from typing import List, Optional, Union
from cemo.tk.tensor.create import create
from cemo.tk import index
Tensor = torch.Tensor
Number = Union[int, float]


def pad(
        x: Tensor,
        dims: List[int],
        new_sizes: List[int],
        fill: Optional[Number] = None,
        requires_grad: bool = False,
        ) -> Tensor:
    """
    Pad a tensor into a new shape by copying the original tensor's values.
    Note: must ensure len(dim) == len(new_sizes)

    Args:
        x: the tensor to be padded
        dims: the dimensions to be padded
        new_sizes: the new sizes of the padded dimensions
        fill: the value to be filled in the new padding area (default: None)

    Returns:
        A tensor of shape new_sizes.
    """
    assert len(dims) == len(new_sizes)
    old_shape = list(x.shape)

    if len(dims) == 0:
        return x
    else:
        idx_old = [slice(old_shape[i]) for i in dims]

    dtype = x.dtype
    device = x.device
    layout = x.layout

    ndim = x.ndim
    std_dims = list(range(ndim))
    dims_pos = [std_dims[i] for i in dims]
    shape_old = list(x.shape)
    size_dict = dict(zip(dims_pos, new_sizes))
    shape_new = [
        size_dict[i] if i in dims_pos else shape_old[i] for i in range(ndim)
    ]

    # create a new empty tensor
    x_new = create(
        shape=shape_new,
        dtype=dtype,
        device=device,
        layout=layout,
        requires_grad=requires_grad,
        fill=fill,
        )

    # copy all values from x to x_full
    idx = index.make_idx(
        ndim=x_new.ndim,
        dims=dims,
        idx=idx_old)
    x_new[idx] = x
    return x_new
