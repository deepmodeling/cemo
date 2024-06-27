import torch
import math
from typing import List, Union, Dict
from cemo.tk.collection import dict_get_item
Tensor = torch.Tensor
FoI = Union[float, int]
DoL = Union[dict, list]
DoT = Union[dict, Tensor]
DT = Dict[str, Tensor]
NDT = Dict[str, Union[DT, Tensor]]  # nested dict of tensors


def even_split(
        dataset: NDT,
        batch_size: int,
        lengths: List[FoI],
        ) -> List[NDT]:
    """
    Split a dataset into a number of subsets evenly.

    Args:
        dataset: a dataset.
        batch_size: size of the dataset.
        lengths: a list of lengths, either counts (int) or fractions (float).

    Returns:
        A list of dicts.
    """
    if type(lengths[0]) is float:
        assert math.fsum(lengths) <= 1.0, \
            "sum of lengths must be <= 1.0, but got {}".format(math.fsum(lengths))    
        cum_ratios = torch.cumsum(torch.tensor(lengths), dim=0)
        cum_ratios[-1] = 1.0  # ensure last ratio is 1.0
        cum_nums = cum_ratios * batch_size
        idx_bounds = [int(math.ceil(x)) for x in cum_nums]
    elif type(lengths[0]) is int:
        assert sum(lengths) == batch_size, \
            f"sum of lengths must be {batch_size}, but got {sum(lengths)}"
        cum_nums = torch.cumsum(torch.tensor(lengths), dim=0)
        idx_bounds = cum_nums.tolist()
    else:
        raise ValueError(f"lengths must be either int or float, but got {type(lengths[0])}")

    if len(idx_bounds) == 1:
        assert idx_bounds[0] == batch_size, \
            f"idx_bounds[0] must be {batch_size}, but got {idx_bounds[0]}"

    def tensor_subsets(d: Tensor, idx_bounds: List[int], accum: List[Tensor]) -> List[Tensor]:
        if len(idx_bounds) == 0:
            return [d]
        elif len(idx_bounds) == 1:
            return accum
        elif len(accum) == 0:
            return tensor_subsets(
                d,
                idx_bounds=idx_bounds,
                accum=[d[0:idx_bounds[0]]],
            )
        else:
            return tensor_subsets(
                d,
                idx_bounds=idx_bounds[1:],
                accum=accum + [d[idx_bounds[0]:idx_bounds[1]]],
            )
    
    def aux_split(d: DoT) -> DoT:
        if type(d) is dict:
            return {k: aux_split(v) for k, v in d.items()}
        elif type(d) is Tensor:
            return tensor_subsets(d, idx_bounds=idx_bounds, accum=[])
        else:
            raise ValueError(f"unsupported type: {type(d)}")
    
    raw_output = aux_split(dataset)

    return [dict_get_item(raw_output, i) for i in range(len(lengths))]
