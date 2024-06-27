from argparse import _SubParsersAction
from cemo.preprocess.subsets import cli_submrcs

def p_submrcs(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the submrcs command.
    """
    p = subparser_set.add_parser(
        "submrcs",
        help="Split mrcs by index files.",
        description="Split mrcs by index files.")
    p.add_argument(
        "-i",
        "--input",
        help="input mrcs file name",
        required=True,
    )
    p.add_argument(
        "-o",
        "--output",
        help="output sub mrcs file name",
        required=True,
    )    
    p.add_argument(
        "--ind",
        help="index file",
        required=True,
    )
    p.set_defaults(func=cli_submrcs)
    return subparser_set