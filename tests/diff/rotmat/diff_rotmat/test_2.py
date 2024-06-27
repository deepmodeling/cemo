import os
import argparse
from cemo.parsers import add_all_parsers


def test():
    f_pkl = os.path.join("..", "data", "mirrored_poses.pkl")
    f_rotmat = os.path.join("..", "data", "mirrored_align_rotmat.txt")
    f_fig = os.path.join("tmp", "test2.png")
    parser = argparse.ArgumentParser()

    _ = add_all_parsers(parser.add_subparsers(dest="subcmd"))
    args = parser.parse_args(
        [
            "diff-rotmat",
            "-i", f_pkl,
            "--align-rotmat", f_rotmat,
            "--fig", f_fig,
            "--fig-title", "test2",
            "--num-bins", "10",
        ]
    )
    args.func(args)
