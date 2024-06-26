"""
Add a subparser for the "cs" subcommand.

author: Yuhang (Steven) Wang
date: 2022/01/12
update: 2022/6/14 restructure the code
"""
from argparse import _SubParsersAction
from cemo.noise.angle import cli_add_angle_noise


def p_add_angle_noise(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add a subparser to subparser_set and return the modified
    subparser_set.
    """
    p = subparser_set.add_parser("add-angle-noise")
    p.add_argument(
        "-i", "--input", required=True,
        help="input cs file")
    p.add_argument(
        "-o", "--output", required=True,
        help="output cs file")
    p.add_argument(
        "--gau-std", type=float, required=True,
        help="standard deviation of the Gaussian noise")
    p.add_argument(
        "--random-seed", type=int, default=123,
        help="default random number generator seed (default: 123)"
    )
    p.set_defaults(func=cli_add_angle_noise)
    return subparser_set
