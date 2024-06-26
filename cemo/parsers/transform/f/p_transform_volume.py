from argparse import _SubParsersAction
from cemo.tk.transform import cli_transform_volume


def p_transform_volume(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Transform a volume (from an mrc file) according to
    a transformation matrix (rotation+translation).
    """
    p = subparser_set.add_parser(
        "transform-volume",
        help="Transform a volume (from an mrc file) according to"
             "a transformation matrix (rotation+translation).",
        description="Transform a volume")
    p.add_argument("-i", "--input", required=True, help="input mrc")
    p.add_argument("-o", "--output", required=True, help="output mrc")
    p.add_argument(
        "-t", "--tmat", required=True,
        help="input rotation+translation matrix txt file (3, 4)")
    p.add_argument(
        "--revert", action="store_true", default=False,
        help="revert the transform")
    p.set_defaults(func=cli_transform_volume)
    return subparser_set
