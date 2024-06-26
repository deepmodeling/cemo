from .split_labels import split_labels
from argparse import Namespace


def cli_split_labels(args: Namespace):
    """
    Split labels to indexs
    """
    split_labels(input_file_name=args.input,output_file_names=args.output,n_types=args.n,start_num=args.startnum)
