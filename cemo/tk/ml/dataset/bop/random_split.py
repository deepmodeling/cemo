import torch
from torch.utils.data import random_split as torch_random_split
from typing import List, Union, Optional, Dict
from cemo.tk.collection import dict_get_item
Tensor = torch.Tensor
FoI = Union[float, int]
DoL = Union[dict, list]
DoT = Union[dict, Tensor]
DT = Dict[str, Tensor]
NDT = Dict[str, Union[DT, Tensor]]  # nested dict of tensors


def random_split(
        dataset: NDT,
        batch_size: int,
        lengths: List[FoI],
        seed: Optional[int] = None,
        debug: bool = False,
        ) -> List[NDT]:
    """
    Divide a dataset dict into a number of subsets,
    where each subset is a dict of tensors of smaller sizes.
    
    Args:
        dataset: a dataset.
        batch_size: size of the dataset.
        lengths: a list of lengths, either counts (int) or fractions (float).
            see: documentation for torch.utils.data.random_split
            https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
        seed: seed for the random number generator.
        debug: if True, print debug information.
    
    Returns:
        A list of dicts.
    """
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        fn_random_split = lambda x: torch_random_split(x, lengths, generator=generator)
    else:
        fn_random_split = lambda x: torch_random_split(x, lengths)

    id_subsets = fn_random_split(torch.arange(batch_size))
    ids_list = [x.indices for x in id_subsets]

    if debug:
        print(f">>> ids_list: {ids_list}")

    def aux_split(d: DoT) -> DoT:
        if type(d) is dict:
            return {k: aux_split(v) for k, v in d.items()}
        elif type(d) is Tensor:
            return [d[ids] for ids in ids_list]
        else:
            raise ValueError(f"unsupported type: {type(d)}")
    
    raw_output = aux_split(dataset)

    return [dict_get_item(raw_output, i) for i in range(len(lengths))]
