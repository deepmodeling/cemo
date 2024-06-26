"""
A data type representing the content of an cryoSPARC cs file

author: Yuhang (Steven) Wang
date: 2022/1/12
update: 2024/1/8 change the hidden data type to torch.Tensor
"""
from dataclasses import dataclass
import numpy
NumpyRecArray = numpy.ndarray


@dataclass
class CryoSparcCS:
    data: NumpyRecArray
    __slots__ = ["data"]
