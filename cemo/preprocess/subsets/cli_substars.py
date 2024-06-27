from .substars import substars
from argparse import Namespace


def cli_substars(args: Namespace):
    """
    Split stars by index files.
    """
    substars(input_file_name=args.input, output_file_name=args.output,index_file_name=args.idx, mrcs_file_name=args.mrcs)
