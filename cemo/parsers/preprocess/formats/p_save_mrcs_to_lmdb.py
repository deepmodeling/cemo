from argparse import _SubParsersAction
from cemo.preprocess.formats import cli_save_mrcs_to_lmdb

def p_save_mrcs_to_lmdb(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the saving mrcs to lmdb command.
    """
    p = subparser_set.add_parser(
        "save-mrcs-to-lmdb",
        help="Convert mrcs file to lmdb format",
        description="Convert mrcs file to lmdb format")
    p.add_argument(
        "-i",
        "--input",
        help="input mrcs file",
        required=True,
    )
    p.add_argument(
        "-o",
        "--output",
        help="output lmdb file",
        required=True,
    )    
    p.add_argument(
        "-c",
        "--chunk",
        default=1000,
        help="chunk size",
        required=False,
    )
    p.set_defaults(func=cli_save_mrcs_to_lmdb)
    return subparser_set
