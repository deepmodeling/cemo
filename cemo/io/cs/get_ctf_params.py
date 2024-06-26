import torch
import logging
import numpy
from typing import Optional
from cemo.io.cs.CryoSparcCS import CryoSparcCS
from cemo.tk.physics.microscope.CTF_Params import CTF_Params
logger = logging.getLogger(__name__)
Tensor = torch.Tensor
NumpyArray = numpy.ndarray


def get_ctf_params(
        cs_obj: CryoSparcCS,
        dtype: torch.dtype = torch.float32) -> CTF_Params:
    """
    Get CTF parameters from a cryoSPARC cs file.

    Args:
        cs_obj: a CryoSparcCS object.
        dtype: The data type of the returned tensors.

    Returns:
        A CTF_Params object
    """

    def to_tensor(x: NumpyArray) -> Tensor:
        return torch.tensor(
            x.copy(),
            dtype=dtype)

    data = cs_obj.data
    return CTF_Params(
        df1_A=to_tensor(data['ctf/df1_A']),
        df2_A=to_tensor(data['ctf/df2_A']),
        df_angle_rad=to_tensor(data['ctf/df_angle_rad']),
        accel_kv=to_tensor(data['ctf/accel_kv']),
        cs_mm=to_tensor(data['ctf/cs_mm']),
        amp_contrast=to_tensor(data['ctf/amp_contrast']),
        phase_shift_rad=to_tensor(data['ctf/phase_shift_rad']),
        bfactor=to_tensor(data['ctf/bfactor']),
    )
