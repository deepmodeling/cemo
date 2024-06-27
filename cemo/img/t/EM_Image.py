from dataclasses import dataclass
from cemo.io.mrc import MRCX
from cemo.io.cs import CryoSparcCS


@dataclass
class EM_Image:
    mrcs: MRCX
    cs: CryoSparcCS
    __slots__ = ["mrcs", "cs"]
