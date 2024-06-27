from argparse import _SubParsersAction
from cemo.img import cli_undo_shift_2d


def p_undo_shift_2d(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the undo shift 2d command.
    """
    p = subparser_set.add_parser(
        "img_undo_shift_2d",
        help="Undo the shift of the image in the x and y direction.",
        description="Undo the shift of the image in the x and y direction.",
    )
    p.add_argument(
        "--in-mrcs",
        help="input mrcs file",
        required=True,
    )
    p.add_argument(
        "--in-cs",
        help="input cs file",
        required=True,
    )
    p.add_argument(
        "--out-mrcs",
        help="output mrcs file",
        required=True,
    )
    p.add_argument(
        "--out-cs",
        help="output cs file",
        required=True,
    )
    p.set_defaults(func=cli_undo_shift_2d)
    return subparser_set
