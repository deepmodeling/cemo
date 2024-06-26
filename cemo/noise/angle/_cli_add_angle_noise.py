from argparse import Namespace
from cemo.io import cs
from cemo.noise.angle import add_noise


def cli_add_angle_noise(args: Namespace):
    """
    Add noise to the projection angles.
    """
    cs.write(
        args.output,
        add_noise(
            cs.read(args.input),
            args.gau_std,
            args.random_seed,
        ),
    )
