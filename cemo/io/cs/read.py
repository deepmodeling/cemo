from cemo.io.cs.CryoSparcCS import CryoSparcCS
import numpy
from typing import Union, Optional, List
from cemo.io.cs.read_index_file import read_index_file
NumpyArray = numpy.ndarray
LInt = List[int]
StrArr = Union[str, NumpyArray, LInt]


def read(
        file_name: str,
        index: Optional[StrArr] = None,
        verbose: bool = False) -> CryoSparcCS:
    """
    Read a cryoSPARC cs file and return 
    a CryoSparcCS object.
    """
    x = numpy.load(file_name)

    if type(index) is str:
        index = read_index_file(index, verbose=verbose)
    elif index is None:
        pass
    elif type(index) is list:
        pass
    elif type(index) is numpy.ndarray:
        pass
    else:
        raise ValueError(f"Unknown index type: {type(index)}")

    if index is not None:
        x = x[index]

    return CryoSparcCS(data=x)
