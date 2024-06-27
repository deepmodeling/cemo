"""
Add Gaussian noise to the angles.

author: Yuhang (Steven) Wang
date: 2022/01/11
"""
from cemo.io.cs import CryoSparcCS
import numpy
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from typing import Optional
from scipy.spatial.transform import Rotation


def add_noise(
        cs: CryoSparcCS,
        std: float,
        seed: Optional[int] = None
        ) -> CryoSparcCS:
    """
    Add Gaussian noise to the angle values of
    each entry.

    Args:
        cs: CryoSparcCS
            Input cryoSPARC cs file.
        std: float
            Standard deviation of the Gaussian noise

    Returns: CryoSparcCS
        A new cryoSPARC object.
    """
    num_items = cs.data.shape[0]

    if seed is None:
        rnd = numpy.random
    else:
        rnd = RandomState(MT19937(SeedSequence(seed)))

    def update_angle(i: int):
        r = Rotation.from_rotvec(cs.data[i]["alignments3D/pose"])
        r_euler = r.as_euler('zyz', degrees=False)
        noise = rnd.normal(loc=0.0, scale=std, size=(3,))
        new_r = Rotation.from_euler('zyz', r_euler + noise, degrees=False)
        new_rotvec = new_r.as_rotvec()
        cs.data[i]["alignments3D/pose"] = new_rotvec

    _ = list(map(update_angle, range(num_items)))

    return cs
