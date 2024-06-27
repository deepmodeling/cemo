import argparse
from cemo.parsers import add_all_parsers


def test():
    f_config = "./config/t1.yml"
    parser = argparse.ArgumentParser()

    _ = add_all_parsers(parser.add_subparsers(dest="subcmd"))
    args = parser.parse_args(
        [
            "fsc",
            "-c", f_config,
        ]
    )
    args.func(args)
