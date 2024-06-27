from argparse import _SubParsersAction
from cemo.preprocess.concat import cli_concatcs

def p_concatcs(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the concatcs command.
    """
    p = subparser_set.add_parser(
        "concatcs",
        help="Concat cs file.",
        description="Concat cs file.")
    p.add_argument(
        "-i",
        "--input",
        help="input cs file name",
        required=True,
    )
    p.add_argument(
        "-o",
        "--output",
        help="output cs file name",
        required=True,
    )
    p.set_defaults(func=cli_concatcs)
    return subparser_set