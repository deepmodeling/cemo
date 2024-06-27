import numpy as np


def make_ctf_params(
        defocusU: float = 12000.,
        defocusV: float = 11000.,
        defocusAngle: float = 80.,
        kV: float = 300.,
        cs: float = 2.7,
        w: float = 0.1,
        phase_shift: float = 0.) -> np.ndarray:

    """
    make CTF parameters

    Args:
        defocusU: max defocus (angstrom)
        defocusV: min defocus (angstrom)
        defocusAngle: defocus angle (degrees)
        kV: voltage (kV)
        cs: spherical aberation (cm)
        w: amplitude contrast
        phase_shift: phase shift (degrees)

    Returns:
        Numpy array of shape (7,)

    """
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
