from .concatmrcs import concat_mrcs
from argparse import Namespace


def cli_concatmrcs(args: Namespace):
    """
    concat cs file from cs downsampled.
    """
    concat_mrcs(input_file_list=args.input, output_file=args.output,apix=args.apix)
