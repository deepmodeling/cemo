import torch
from .undo_shift_2d import undo_shift_2d
from argparse import Namespace
from cemo.io import mrc, cs
from ..t import EM_Image


def cli_undo_shift_2d(args: Namespace):
    """
    Undo 2D shifts of the images.

    Args:
        args: The arguments from the command line.
    """
    obj_mrcs = mrc.read(args.in_mrcs)
    obj_cs = cs.read(args.in_cs)
    obj_img = EM_Image(
        mrcs=obj_mrcs,
        cs=obj_cs,
    )
    new_obj_img = undo_shift_2d(obj_img)
    cs.write(args.out_cs, new_obj_img.cs)
    mrc.write(args.out_mrcs, new_obj_img.mrcs)
