from argparse import _SubParsersAction
from cemo.preprocess.concat import cli_concatmrcs

def p_concatmrcs(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the concatcs command.
    """
    p = subparser_set.add_parser(
        "concatmrcs",
        help="Concat mrcs file.",
        description="Concat mrcs file.")
    p.add_argument(
        "-i",
        "--input",
        help="input mrcs file name",
        required=True,
    )
    p.add_argument(
        "-o",
        "--output",
        help="output mrcs file name",
        required=True,
    )
    
    p.add_argument(
        "--apix",
        help="apix value",
        required=True,
    )
    p.set_defaults(func=cli_concatmrcs)
    return subparser_set