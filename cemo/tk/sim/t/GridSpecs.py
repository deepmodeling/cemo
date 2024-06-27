from dataclasses import dataclass
from typing import Tuple
from torch import Tensor


@dataclass
class GridSpecs:
    shape: Tuple[int, int, int]
    apix: float  # angstroms per voxel
    center: Tensor  # grid center coordinates
    res: float  # target resolution
