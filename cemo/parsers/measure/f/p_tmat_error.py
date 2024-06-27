from argparse import _SubParsersAction
from cemo.measure import cli_tmat_error


def p_tmat_error(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Calculate the rotmat prediction errors.
    """
    p = subparser_set.add_parser(
        "tmat-error",
        help="Calculate the rotation/shift prediction errors",
        description="Calculate the rotation/shift prediction errors")
    p.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="config file")
    p.add_argument(
        "--skip-volume-align",
        action="store_true",
        help="skip volume alignment")
    p.add_argument(
        "--rotation-error",
        action="store_true",
        help="Do rotation error calculation"
    )
    p.add_argument(
        "--rotation-error-squared",
        action="store_true",
        help="Return squared rotation error"
    )
    p.add_argument(
        "--rotation-error-stat",
        default="unspecified",
        choices=["mean", "median"],
        help="Rotation error statistic type (mean or median)"
    )
    p.add_argument(
        "--shift-error",
        action="store_true",
        help="Do shift error calculation"
    )
    p.add_argument(
        "--shift-error-squared",
        action="store_true",
        help="Return squared shift error"
    )
    p.add_argument(
        "--shift-error-stat",
        choices=["mean", "median"],
        default="unspecified",
        help="Shift error statistic type: mean or median"
    )
    p.add_argument(
        "--shift-use-rotation",
        action="store_true",
        help="Apply 3D rotation to the 2D shift vector"
    )
    
    p.set_defaults(func=cli_tmat_error)
    return subparser_set
