import numpy


def calc_fsc_res(
        fsc_data: numpy.ndarray,
        gold_std: float = 0.143) -> numpy.ndarray:
    """
    Get FSC

    Args:
        fsc_data: an array of shape (N, 2). 
            The first column is the fsc resolution (must be in ascending order)
            The second column is the FSC correlation coefficient.
        gold_std: Gold standard value
    Returns:
        the resolution for the gold_std
    """
    # find the index where y < 0.2
    fsc_cutoff = gold_std + 0.05
    if fsc_data[0, 1] > fsc_cutoff:
        res = fsc_data[0, 0]
    else:
        idx = numpy.searchsorted(fsc_data[:, 1], fsc_cutoff, side="left")
        x, y = fsc_data[:idx, 1], fsc_data[:idx, 0]
        res = numpy.interp(gold_std, x, y)
    return res
