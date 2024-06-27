from argparse import _SubParsersAction
from cemo.img import cli_random_shift_2d


def p_random_shift_2d(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the random shift 2d command.
    """
    p = subparser_set.add_parser(
        "img_random_shift_2d",
        help="Randomly shift the image in the x and y direction.",
        description="Randomly shift the image in the x and y direction.",
    )
    p.add_argument(
        "--dist",
        type=str,
        help="The distribution to use for the shift.",
        choices=["uniform", "gaussian"],
        default="uniform",
    )
    p.add_argument(
        "--in-mrcs",
        help="input mrcs file",
        required=True,
    )
    p.add_argument(
        "--in-cs",
        help="input cs file",
        required=True,
    )
    p.add_argument(
        "--out-mrcs",
        help="output mrcs file",
        required=True,
    )
    p.add_argument(
        "--out-cs",
        help="output cs file",
        required=True,
    )
    p.add_argument(
        "--x-shift-percent",
        type=float,
        required=True,
        help="The max percent of image width allowed to shift",
    )
    p.add_argument(
        "--y-shift-percent",
        type=float,
        required=True,
        help="the max percent of image height allowed to shift",
    )
    p.add_argument(
        "--x-std-percent",
        type=float,
        help="The standard deviation of the Gaussian distribution along x"
             "in percentage of image width",
        default=0.05,
    )
    p.add_argument(
        "--y-std-percent",
        type=float,
        help="The standard deviation of the Gaussian distribution along y"
             "in percentage of image height",
        default=0.05,
    )
    p.set_defaults(func=cli_random_shift_2d)
    return subparser_set
