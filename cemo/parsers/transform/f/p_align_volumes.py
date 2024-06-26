from argparse import _SubParsersAction
from cemo.tk.transform import cli_align_volumes


def p_align_volumes(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Align two volume files (mrc).
    """
    p = subparser_set.add_parser(
        "align-volumes",
        help="Align two volume files (mrc).",
        description="Align two volume files (mrc).")
    p.add_argument("-c", "--config", required=True, help="config file")
    p.set_defaults(func=cli_align_volumes)
    return subparser_set
