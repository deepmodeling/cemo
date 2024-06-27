import os
import numpy
import logging
from typing import List, Union
from cemo.io import pkl
logger = logging.getLogger(__name__)
NumpyArray = numpy.ndarray
LInt = List[int]
LIntArr = Union[LInt, NumpyArray]


def read_index_file(file_name: str, verbose: bool = False) -> LIntArr:
    """
    Read a numpy file containing a list of indices.

    Args:
        file_name: The name of the file.
        verbose: Whether to print log messages.

    Returns:
        A tensor containing the indices.
    """
    ext = os.path.splitext(file_name)[1]

    if verbose:
        logger.info(f"Reading index file {file_name}...")

    if ext == ".pkl":
        return pkl.read(file_name)
    elif ext == ".npy":
        return numpy.load(file_name)
    elif ext == ".txt":
        return numpy.loadtxt(file_name)
    else:
        raise ValueError(f"Unknown index file extension: {ext}")
