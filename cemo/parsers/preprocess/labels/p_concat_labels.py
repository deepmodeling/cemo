from argparse import _SubParsersAction
from cemo.preprocess.labels import cli_concat_labels

def p_concat_labels(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the label split command.
    """
    p = subparser_set.add_parser(
        "concat-label",
        help="Concat indexs",
        description="Concat indexs")
    p.add_argument(
        "-i",
        "--input",
        help="input index file names, need to be %d format",
        required=True,
    )
    p.add_argument(
        "-o",
        "--output",
        help="output index file name",
        required=True,
    )    
    p.add_argument(
        "-n",
        "--num",
        help="nums, requires a list format",
        required=True,
    )
    p.set_defaults(func=cli_concat_labels)
    return subparser_set