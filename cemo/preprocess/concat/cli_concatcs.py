from .concatcs import concat_cs
from argparse import Namespace


def cli_concatcs(args: Namespace):
    """
    concat cs file from cs downsampled.
    """
    concat_cs(input_file_list=args.input, output_file=args.output)
