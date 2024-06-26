import numpy


def reprocess_fsc(fsc: numpy.array):
    """
    Reprocess the FSC data.
    """
    x = fsc.copy()
    x[:, 0] = 1.0 / x[:, 0]
    return numpy.flip(x, axis=0)
