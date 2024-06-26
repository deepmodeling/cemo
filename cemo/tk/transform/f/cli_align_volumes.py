from argparse import Namespace
from cemo.io import yml
from .align_volumes import align_volumes


def cli_align_volumes(args: Namespace):
    config = yml.read(args.config)
    _ = [align_volumes(c, config["env"]) for c in config["files"]]
