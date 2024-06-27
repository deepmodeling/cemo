import torch
from .uniform_shift_2d import uniform_shift_2d
from .gaussian_shift_2d import gaussian_shift_2d
from argparse import Namespace
from cemo.io import mrc, cs
from ..t import EM_Image


def cli_random_shift_2d(args: Namespace):
    """
    Shift the image randomly.

    Args:
        args: The arguments from the command line.
    """
    obj_mrcs = mrc.read(args.in_mrcs)
    obj_cs = cs.read(args.in_cs)
    obj_img = EM_Image(
        mrcs=obj_mrcs,
        cs=obj_cs,
    )
    shift_percent = torch.tensor([args.x_shift_percent, args.y_shift_percent])
    if args.dist == "uniform":
        new_obj_img = uniform_shift_2d(obj_img, shift_percent)
    elif args.dist == "gaussian":
        shift_std = torch.tensor([args.x_std_percent, args.y_std_percent])
        new_obj_img = gaussian_shift_2d(obj_img, shift_std, shift_percent)
    else:
        raise ValueError(f"Unknown distribution: {args.dist}")
    cs.write(args.out_cs, new_obj_img.cs)
    mrc.write(args.out_mrcs, new_obj_img.mrcs)
