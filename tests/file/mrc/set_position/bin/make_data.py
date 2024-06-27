import mrcfile
import numpy


def main():
    f_out = "data/t1.mrc"
    with mrcfile.new(f_out, overwrite=True) as OUT:
        OUT.set_data(numpy.zeros((2, 2, 2), dtype=numpy.float32))
        OUT.header.origin = (1., 2., 3.)
        OUT.header.nxstart = 10
        OUT.header.nystart = 20
        OUT.header.nzstart = 30


if __name__ == "__main__":
    main()
