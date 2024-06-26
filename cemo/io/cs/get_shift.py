import numpy
import torch
import logging
from typing import Optional
from cemo.io.cs.CryoSparcCS import CryoSparcCS
logger = logging.getLogger(__name__)
NumpyRecArray = numpy.ndarray
NumpyArray = numpy.ndarray
Tensor = torch.Tensor


def get_shift(
        cs_obj: CryoSparcCS,
        is_abinit: bool,
        return_ratio: bool,
        blob_shape: Optional[NumpyArray] = None,
        verbose: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        ) -> Tensor:
    """
    Get rotation matrix and shift vectors from a cryoSPARC cs file.

    Args:
        cs_obj: a CryoSparcCS object.
        CryoSparcCS object.
        is_abinit: Whether the cs file is from ab-initio reconstruction.
        return_ratio: Whether to return shift ratio.
        dtype: The data type of the returned tensors.
        verbose: Whether to print log messages.

    Returns:
        A tuple containing the shift ratio vectors.
    """
    if type(cs_obj) is not CryoSparcCS:
        raise TypeError(
            f"cs_obj must be a CryoSparcCS object, got {type(cs_obj)}")
    cs_data = cs_obj.data

    if is_abinit:
        TKEY = "alignments_class_0/shift"
    else:
        TKEY = "alignments3D/shift"

    # Parse translations
    if verbose:
        logger.info(f"Extracting translations from {TKEY} ...")

    # convert translations from pixels to ratio
    shift_px = cs_data[TKEY]
    if return_ratio:
        if blob_shape is None:
            assert "blob/shape" in cs_data.dtype.names
            blob_shape = cs_data["blob/shape"].astype(shift_px.dtype)
        else:
            blob_shape = blob_shape.astype(shift_px.dtype)
        shape = cs_data["blob/shape"].astype(shift_px.dtype)  # (N, 2)
        shift = torch.tensor(
            shift_px/shape,
            dtype=dtype,
            device=device)
    else:
        # note: must use "shift_px.copy()""to return a numpy array
        # with contiguous memory layout, otherwise the returned
        # tensor will be a view of the original numpy array.
        # Since cs_data is a record array and has a complex memory layout,
        # torch.tensor(cs_data[TKEY]) will raise an error:
        #    ValueError: given numpy array strides not a multiple
        #                of the element byte size.
        #                Copy the numpy array to reallocate the memory
        shift = torch.tensor(
            shift_px.copy(),
            dtype=dtype,
            device=device)

    return shift
