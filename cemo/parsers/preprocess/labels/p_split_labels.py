from argparse import _SubParsersAction
from cemo.preprocess.labels import cli_split_labels

def p_split_labels(subparser_set: _SubParsersAction) -> _SubParsersAction:
    """
    Add the subparser for the label split command.
    """
    p = subparser_set.add_parser(
        "split-label",
        help="Split index from labels",
        description="Split index from labels")
    p.add_argument(
        "-i",
        "--input",
        help="input labels file name",
        required=True,
    )
    p.add_argument(
        "-o",
        "--output",
        help="output index file names, need to be %d format",
        required=True,
    )    
    p.add_argument(
        "-n",
        "--num",
        default=10,
        help="number of classes",
        required=False,
    )
    p.add_argument(
        "--start-num",
        default=0,
        help="start number",
        required=False,
    )
    p.set_defaults(func=cli_split_labels)
    return subparser_set