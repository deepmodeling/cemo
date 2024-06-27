import argparse
from cemo.parsers import add_all_parsers
from cemo.io import mrc


def test():
    f_in = "data/t1.mrc"
    f_out = "tmp/t1.mrc"
    parser = argparse.ArgumentParser()

    _ = add_all_parsers(parser.add_subparsers(dest="subcmd"))
    args = parser.parse_args(
        [
            "file-mrc-set-location",
            "-i", f_in,
            "-o", f_out,
            "-ox", "0.0",
            "-oy", "0.0",
            "-oz", "0.0",
            "-nx", "0.0",
            "-ny", "0.0",
            "-nz", "1.0",
        ]
    )
    args.func(args)
    m = mrc.read(f_out)
    assert m.header.origin.x == 0.
    assert m.header.origin.y == 0.
    assert m.header.origin.z == 0.
    assert m.header.nxstart == 0.
    assert m.header.nystart == 0.
    assert m.header.nzstart == 1.
