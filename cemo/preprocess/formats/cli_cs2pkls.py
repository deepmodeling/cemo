from .cs2pkls import cs2pkls
from argparse import Namespace


def cli_cs2pkls(args: Namespace):
    """
    Convert cs to pkls
    """
    cs2pkls(input_file_name=args.input,output_ctf_name=args.ctf,output_pose_name=args.pose)
