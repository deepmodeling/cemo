from argparse import _SubParsersAction
from cemo.measure import cli_fsc


def p_fsc(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Calculate the Fourier shell correlation
    between two volume files (mrc).
    """
    p = subparser_set.add_parser(
        "fsc",
        help="Calculate the Fourier shell correlation"
             "between two volume files (mrc).",
        description="Calculate the Fourier shell correlation"
                    "between two volume files (mrc).")
    p.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="config file")
    p.set_defaults(func=cli_fsc)
    return subparser_set
