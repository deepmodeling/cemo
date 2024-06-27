from .submrcs import submrcs
from argparse import Namespace


def cli_submrcs(args: Namespace):
    """
    Split mrcs by index files.
    """
    submrcs(input_mrcs_file_name=args.input, output_mrcs_file_name=args.output,index_file_name=args.idx)
