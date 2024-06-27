from argparse import Namespace
from cemo.io import mrc, tensor
import torch
from .transform_volumes import transform_volumes


def cli_transform_volume(args: Namespace):
    """
    Transform a volume (from an mrc file) according to
    a transformation matrix (rotation+translation).

    Args:
        args: Namespace object containing the command line arguments.
            args.input is the input mrc file.
            args.tmat is the transformation matrix (txt) file (shape: 3, 4),
                where the first three columns represents the rotation and 
                the last column represents the translation.
            args.revert: if true, do the reverse transformation,
                i.e., if the forward transformation is a1 = R12*a2 + t12,
                the reverse transformation is a2 = R12^T * (a1 - t12), where
                R12 is the rotation matrix and t12 is the translation vector.
    """
    m = mrc.read(args.input)
    dtype = torch.float32
    volume_in = torch.tensor(m.data, dtype=dtype)
    T = tensor.read(args.tmat, dtype=dtype)
    if args.revert:  # do the reverse transformation
        rotmat = T[:, :3].transpose(0, 1)  # (3, 3)
        shift = -(rotmat @ T[:, -1])  # (3, 1)
    else:
        rotmat = T[:, :3]  # (3, 3)
        shift = T[:, -1]   # (3, 1)

    batch_rotmat = rotmat.unsqueeze(dim=0)  # (1, 3, 3)
    batch_shift = shift.unsqueeze(dim=0).unsqueeze(dim=-1)  # (1, 3, 1)
    volume_new = transform_volumes(
        volume_in,
        batch_rotmat,
        batch_shift,
        )
    m.data = volume_new.squeeze(dim=0).numpy()
    mrc.write(args.output, m)
