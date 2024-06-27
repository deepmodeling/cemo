from argparse import _SubParsersAction
from cemo.preprocess.formats import cli_cs2pkls

def p_cs2pkls(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the converting cs to pkl command.
    """
    p = subparser_set.add_parser(
        "cs2pkl",
        help="Convert cs file to pkl format for cryodrgn usage",
        description="Convert cs file to pkl format for cryodrgn usage")
    p.add_argument(
        "-i",
        "--input",
        help="input cs file",
        required=True,
    )
    p.add_argument(
        "--ctf",
        help="output ctf pkl file",
        required=True,
    )    
    p.add_argument(
        "--pose",
        help="output pose pkl file",
        required=True,
    )    
    p.set_defaults(func=cli_cs2pkls)
    return subparser_set
