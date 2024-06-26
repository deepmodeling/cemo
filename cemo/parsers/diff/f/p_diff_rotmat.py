from argparse import _SubParsersAction
from cemo.diff.rotmat import cli_diff_rotmat


def p_diff_rotmat(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the random shift 2d command.
    """
    p = subparser_set.add_parser(
        "diff-rotmat",
        help="Calculate the difference between two rotation matrices",
        description="Calculate the difference between two rotation matrices")
    p.add_argument(
        "-i",
        "--input",
        help="input pkl file which containts the original and predicted"
             "rotation matrices",
        required=True,
    )
    p.add_argument(
        "-o",
        "--output",
        default="",
        help="output data file which contains the rotation matrix difference",
        required=False,
    )
    p.add_argument(
        "--align-rotmat",
        help="input rotation matrix text file for coordinate system alignment",
        required=True,
    )
    p.add_argument(
        "--num-bins",
        type=int,
        help="number of bins for the histogram",
        default=10,
    )
    p.add_argument(
        "--mirror-rotmat",
        default="",
        help="Mirror-transformation matrix txt file",
    )
    p.add_argument(
        "--squared",
        action="store_true",
        help="If set, return the squared rotation matrice difference"
    )
    p.add_argument(
        "--stat",
        choices=["mean", "median"],
        required=True,
        help="Rotation matrix error statistic type (mean or median)"
    )
    p.add_argument(
        "--fig",
        help="output figure file",
        required=True,
    )
    p.add_argument(
        "--fig-title",
        default="Rotation matrix difference",
        help="title of the figure",
    )
    p.add_argument(
        "--x-label",
        default="log10(error)",
        help="figure x-label",
    )
    p.add_argument(
        "--y-label",
        default="count",
        help="figure y-label",
    )
    p.set_defaults(func=cli_diff_rotmat)
    return subparser_set
