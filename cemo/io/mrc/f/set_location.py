from .read import read as read_mrc
from .write import write as write_mrc
from argparse import Namespace


def set_location(args: Namespace):
    """
    Set the location of a MRC/MRCS coordinate system.
    """
    m = read_mrc(args.input)
    m.header.origin.x = args.origin_x
    m.header.origin.y = args.origin_y
    m.header.origin.z = args.origin_z
    m.header.nxstart = args.nxstart
    m.header.nystart = args.nystart
    m.header.nzstart = args.nzstart
    print("new origin:", m.header.origin)
    print("new nxstart: ", m.header.nxstart)
    print("new nystart: ", m.header.nystart)
    print("new nzstart: ", m.header.nzstart)
    write_mrc(args.output, m)
