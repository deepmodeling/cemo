from .save_mrcs_to_lmdb import save_mrcs_to_lmdb
from argparse import Namespace


def cli_save_mrcs_to_lmdb(args: Namespace):
    """
    Convert mrcs to lmdb
    """
    save_mrcs_to_lmdb(f_mrcs=args.input,f_output=args.output,chunk=args.chunk)
