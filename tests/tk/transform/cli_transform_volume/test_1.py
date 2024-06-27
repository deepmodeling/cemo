import argparse
from cemo.parsers import add_all_parsers


def test():
    f_in = "tmp/benchmark/6vyb/6vyb_snr2.0_no-ctf_10k_volume_new_origin.mrc"
    f_out = "tmp/benchmark/6vyb/6vyb_snr2.0_no-ctf_10k_volume_new_origin_flipz_tmp.mrc"
    f_tmat = "./data/flipz_tmat.txt"
    parser = argparse.ArgumentParser()

    _ = add_all_parsers(parser.add_subparsers(dest="subcmd"))
    args = parser.parse_args(
        [
            "transform-volume",
            "-i", f_in,
            "-o", f_out,
            "-t", f_tmat,
        ]
    )
    args.func(args)
    print(f_out)
