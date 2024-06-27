from cemo.io.cs import CryoSparcCS
import numpy


def write(f_out: str, cs: CryoSparcCS):
    """
    Write an cryoSPARC cs file.

    Args:
        f_out: str
            Output file name
        cs: CryoSparcCS
            Data to be writen out
    """
    with open(f_out, "wb") as OUT:
        numpy.save(OUT, cs.data)
