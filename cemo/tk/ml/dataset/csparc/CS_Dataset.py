import torch
import os
from torch.utils.data import Dataset
import logging
from typing import Optional, Tuple
import cemo.io.cs as cs
import numpy
# from hetem.tk import make_odd
logger = logging.getLogger(__name__)
Tensor = torch.Tensor


class CS_Dataset(Dataset):
    """
    Convert a cryoSPARC cs file into a PyTorch Dataset.
    """
    def __init__(
            self,
            file_path: str,
            image_size: int,
            use_shift_ratio: bool,
            is_abinit: bool = False,
            dtype: torch.dtype = torch.float32,
            use_ctf: bool = False,
            index_file: Optional[str] = None,
            verbose: bool = False,
            ):
        if verbose:
            logger.info(f"Loading {file_path}...")

        # check file_path exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found")

        cs_obj = cs.read(file_path, index=index_file, verbose=verbose)
        cs_data = cs_obj.data

        self.length = len(cs_data)
        if verbose:
            logger.info(f"Dataset size: {self.length}")

        if verbose:
            logger.info(f"{self.length} particles loaded in `{file_path}`")
        
        # image size
        self.image_size = image_size

        # rotation matrices
        self.rotmat = cs.get_rotmat(
            cs_obj, is_abinit=is_abinit, dtype=dtype, verbose=verbose)

        # shift vectors
        self.shift = cs.get_shift(
            cs_obj,
            is_abinit=is_abinit,
            return_ratio=use_shift_ratio,
            dtype=dtype, verbose=verbose)

        # CTF
        self.use_ctf = use_ctf
        if use_ctf:
            self.ctf_params = cs.get_ctf_params(
                cs_obj, dtype=torch.float32)
        else:
            self.ctf_params = None

        # particle blob indices & paths
        self.blob_idx = numpy.array(cs_data["blob/idx"], dtype=numpy.int32)
        self.blob_path = cs_data["blob/path"].astype(str).tolist()
        # remove prefix ">"
        if self.blob_path[0].startswith(">"):
            self.blob_path = [p[1:] for p in self.blob_path]

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        if self.use_ctf:
            ctf_params = self.ctf_params[idx]
        else:
            ctf_params = torch.tensor([], dtype=torch.float32)

        return (
            self.rotmat[idx],
            self.shift[idx],
            ctf_params,
        )
