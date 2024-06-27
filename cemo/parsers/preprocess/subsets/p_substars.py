from argparse import _SubParsersAction
from cemo.preprocess.subsets import cli_substars

def p_substars(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the substars command.
    """
    p = subparser_set.add_parser(
        "substars",
        help="Split stars by index files.",
        description="Split stars by index files.")
    p.add_argument(
        "-i",
        "--input",
        help="input stars file name",
        required=True,
    )
    p.add_argument(
        "-o",
        "--output",
        help="output sub stars file name",
        required=True,
    )
    p.add_argument(
        "--ind",
        help="index file",
        required=True,
    )
    p.add_argument(
        "--mrcs",
        help="mrcs file for replacing references",
        required=True,
    )
    p.set_defaults(func=cli_substars)
    return subparser_set