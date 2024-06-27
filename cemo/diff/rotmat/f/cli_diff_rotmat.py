from .diff_rotmat import diff_rotmat
import numpy
from argparse import Namespace


def cli_diff_rotmat(args: Namespace) -> numpy.array:
    """
    Add noise to the projection angles.

    Input pkl data file must contain a dictionary with the following keys:
        "rot": ground-truth rotation matrices
        "real_rot_pred": best predicted rotation matrices
    """
    diff_rotmat(
        f_data=args.input,
        f_align_rotmat=args.align_rotmat,
        squared=args.squared,
        stat=args.stat,
        f_fig=args.fig,
        num_bins=args.num_bins,
        fig_title=args.fig_title,
        xlabel=args.x_label,
        ylabel=args.y_label,
        f_mirror_rotmat=args.mirror_rotmat,
        f_data_out=args.output,
    )
