"""
Read MRCS files.

author: Yuhang Wang
date: 2020.4.22
"""
from ..t.MRCX import MRCX
from mrcfile.mrcfile import MrcFile


def read(
        file_name: str,
        ) -> MRCX:
    """
    Read MRCS files.

    Args:
        file_name: str
            The file_name of the MRCS file.

    Returns: MRCX
        The MRCX object.
    """
    with MrcFile(file_name, "r", permissive=True) as mrc:
        return MRCX(
            voxel_size=mrc.voxel_size.copy(),
            header=mrc.header.copy(),
            ext_header=mrc.extended_header.copy(),
            data=mrc.data.copy(),
        )
