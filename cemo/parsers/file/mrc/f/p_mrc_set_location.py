from argparse import _SubParsersAction
from email.policy import default
from cemo.io import mrc


def p_mrc_set_location(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the random shift 2d command.
    """
    p = subparser_set.add_parser(
        "file-mrc-set-location",
        help="Set the location of a MRC coordinate system",
        description="Set the location of a MRC coordinate system")
    p.add_argument(
        "-i",
        "--input",
        help="input MRC file",
        required=True,
    )
    p.add_argument(
        "-o",
        "--output",
        help="output MRC file",
        required=True,
    )
    p.add_argument(
        "-ox", "--origin-x", default=0., type=float,
        help="x of the new origin")
    p.add_argument(
        "-oy", "--origin-y", default=0., type=float,
        help="y of the new origin")
    p.add_argument(
        "-oz", "--origin-z", default=0., type=float,
        help="z of the new origin")
    p.add_argument(
        "-nx", "--nxstart", default=0., type=float,
        help="nxstart of the new origin")
    p.add_argument(
        "-ny", "--nystart", default=0., type=float,
        help="nystart of the new origin")
    p.add_argument(
        "-nz", "--nzstart", default=0., type=float,
        help="nzstart of the new origin")
    p.set_defaults(func=mrc.set_location)
    return subparser_set
