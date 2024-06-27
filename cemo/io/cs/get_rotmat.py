import numpy
import torch
from scipy.spatial.transform import Rotation
import logging
from cemo.io.cs.CryoSparcCS import CryoSparcCS
logger = logging.getLogger(__name__)
NumpyRecArray = numpy.ndarray
NumpyArray = numpy.ndarray
Tensor = torch.Tensor


def get_rotmat(
        cs_obj: CryoSparcCS,
        is_abinit: bool,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        ) -> Tensor:
    """
    Get rotation matrix and shift vectors from a cryoSPARC cs file.

    Args:
        cs_obj: a CryoSparcCS object.
        is_abinit: Whether the cs file is from ab-initio reconstruction.
        dtype: The data type of the returned tensors.
        verbose: Whether to print log messages.

    Returns:
        A tuple containing the rotation matrix
    """
    if type(cs_obj) is not CryoSparcCS:
        raise TypeError(
            f"cs_obj must be a CryoSparcCS object, got {type(cs_obj)}")
    cs_data = cs_obj.data

    if is_abinit:
        RKEY = "alignments_class_0/pose"
    else:
        RKEY = "alignments3D/pose"

    # Parse rotations
    if verbose:
        logger.info(f"Extracting rotations from `{RKEY}` ...")
    rotvec = cs_data[RKEY].copy()
    rotmat_np = Rotation.from_rotvec(rotvec)
    rotmat = torch.tensor(rotmat_np.as_matrix(), dtype=dtype)  # (N, 3, 3)

    return rotmat
