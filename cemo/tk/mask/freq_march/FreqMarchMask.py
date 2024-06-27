from dataclasses import dataclass
from typing import Callable
FnScheduler = Callable[[int], float]


@dataclass
class FreqMarchMask:
    r_min: int
    r_max: int
    r_now: int
    mask_shape: str
    scheduler: FnScheduler
