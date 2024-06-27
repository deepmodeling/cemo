from dataclasses import dataclass
from torch import Tensor

@dataclass
class AtomTraj:
    frames: Tensor
    radii: Tensor
    mass: Tensor
