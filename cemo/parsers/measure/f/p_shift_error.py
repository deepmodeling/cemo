from argparse import _SubParsersAction
from cemo.measure import cli_shift_error


def p_shift_error(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Calculate the prediction error for 2D shifts.
    """
    p = subparser_set.add_parser(
        "shift-error",
        help="Calculate the shift prediction errors",
        description="Calculate the shift prediction errors")
    p.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="config file")
    p.add_argument(
        "--invert-rotmat",
        action="store_true",
        help="invert the rotation matrix")
    p.set_defaults(func=cli_shift_error)
    return subparser_set
