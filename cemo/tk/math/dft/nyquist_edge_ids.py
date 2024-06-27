from typing import List


def nyquist_edge_ids(N: int, symm: bool) -> List[int]:
    """
    Find the index for the Nyquist frequency position
    for a 1D Fourier frequency array.

    Args:
        N: size of the Fourier frequency array
        symm: whether the input frequency matrix x's layout is symmetric

    Returns:
        return [0] if symm is True, otherwise return [N//2]
    """
    if symm:
        output = [0]
    else:
        output = [N//2]

    return output
