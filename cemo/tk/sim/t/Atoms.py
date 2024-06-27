from dataclasses import dataclass
from torch import Tensor

@dataclass
class Atoms:
    coords: Tensor
    radii: Tensor
    mass: Tensor
