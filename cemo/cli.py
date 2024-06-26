"""
mrco: MRC file operations.

author: Yuhang(Steven) Wang
date: 2022/01/05
update: 2022/06/14 restructure the code
"""
import argparse
from argparse import Namespace
from cemo.parsers import add_all_parsers


def parse_args(print_help: bool = False) -> Namespace:
    parser = argparse.ArgumentParser(
        description="CEMO")
    _ = add_all_parsers(parser.add_subparsers(dest="subcmd"))
    if print_help:
        parser.parse_args(["--help"])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.subcmd is None: 
        parse_args(print_help=True)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
