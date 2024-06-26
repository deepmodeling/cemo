import numpy as np


def make_ctf_params_uniform(
        defocusMean_min: float = 10000.,
        defocusMean_max: float = 20000.,
        defocusGap_min: float = 100.,
        defocusGap_max: float = 1000.,
        defocusAngle_min: float = 10.,
        defocusAngle_max: float = 80.,
        kV: float = 300.,
        cs: float = 2.7,
        w: float = 0.1,
        phase_shift: float = 0.) -> np.ndarray:
    """
    Make a CTF with uniformly distributed parameters.

    """
    defocusMean = np.random.uniform(low=defocusMean_min, high=defocusMean_max)
    gap = np.random.uniform(low=defocusGap_min, high=defocusGap_max)
    defocusU = defocusMean - gap/2.
    defocusV = defocusMean + gap/2.
    defocusAngle = np.random.uniform(low=defocusAngle_min, high=defocusAngle_max)
    return np.array(
        [
            defocusU,
            defocusV,
            defocusAngle,
            kV,
            cs,
            w,
            phase_shift,
        ]
    )
