"""
Make an MRC file.

author: Yuhang (Steven) Wang
date: 2022/01/05
"""
import argparse
import numpy 
from argparse import Namespace
import mrcfile


def get_args() -> Namespace:
    p = argparse.ArgumentParser("Make an MRC file")
    p.add_argument("-x", "--x-len", required=True, type=int,
        help="number of grid points along x")
    p.add_argument("-y", "--y-len", required=True, type=int,
        help="number of grid points along y")
    p.add_argument("-z", "--z-len", required=True, type=int,
        help="number of grid points along z")
    p.add_argument("-o", "--output", required=True,
        help="output file")
    return p.parse_args()

def main():
    args = get_args()
    nx, ny, nz = args.x_len, args.y_len, args.z_len
    data_shape = (nx, ny, nz)
    data = numpy.random.random_sample(data_shape).astype(numpy.float32)
    with mrcfile.new(args.output, overwrite=True) as mrc:
        mrc.set_data(data)
        mrc.header.cella = (float(nx), float(ny), float(nz))
        print("new mrc header", mrc.header)
        print(mrc.header.dtype)
    if mrcfile.validate(args.output) is not True:
        print(f"Error: couldn't create the output file {args.output}")
    print(args.output)
    print(data_shape)
    print(data.shape)
    print
    

if __name__ == "__main__":
    main()
