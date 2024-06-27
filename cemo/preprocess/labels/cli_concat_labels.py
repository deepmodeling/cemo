from .concat_labels import concat_labels
from argparse import Namespace


def cli_concat_labels(args: Namespace):
    """
    Split labels to indexs
    """
    concat_labels(input_file_names=args.input,output_file_name=args.output,nums=args.num)
