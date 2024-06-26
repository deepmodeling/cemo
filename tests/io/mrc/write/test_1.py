from cemo.io import mrc
import os


def test():
    f_in = os.path.join("..", "data", "tetra_snr0.1_2.mrcs")
    f_out = os.path.join("tmp", "out1.mrcs")
    obj_in = mrc.read(f_in)

    if os.path.exists(f_out):
        os.remove(f_out)
    mrc.write(f_out, obj_in)
