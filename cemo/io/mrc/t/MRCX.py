"""
Representing the MRC/MRCS data type
author: Yuhang Wang
date: 2022.7.18
"""
import numpy
from dataclasses import dataclass
from typing import Optional


@dataclass
class MRCX:
    """
    MRCS data type
    """
    voxel_size: float = 1.0
    header: Optional[numpy.recarray] = None
    ext_header: Optional[numpy.recarray] = None
    data: Optional[numpy.ndarray] = None
