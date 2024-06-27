from argparse import Namespace
from cemo.io import yml
from .calc_fsc import calc_fsc


def cli_fsc(args: Namespace):
    config = yml.read(args.config)
    _ = [calc_fsc(c, config["env"]) for c in config["files"]]
