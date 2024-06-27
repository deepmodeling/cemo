from typing import List


def fourier_edge_ids(N: int, symm: bool) -> List[int]:
    """
    Find the indices for the lower half of the edge elements with
    Fourier coefficient symmetry.
    If N is even, the edge column indices are [0, N//2], otherwise [i_DC].
    If symm = True, i_DC = N//2, otherwise i_DC = 0.

    Args:
        N: image size
        symm: whether the input coefficient matrix x's layout is symmetric

    Returns:
        A list of edge indices
    """
    i_mid = N // 2
    if symm:
        i_DC = i_mid
    else:
        i_DC = 0

    if N % 2 == 0:
        edges = [0, i_mid]
    else:
        edges = [i_DC]

    return edges
