import torch
from torch import Tensor
from typing import Optional


def rotmat_diff(
        rotmat_ref: Tensor,
        rotmat_target: Tensor,
        squared: bool,
        align_matrix: Optional[Tensor] = None,
        ord: str = "fro",
        ) -> Tensor:
    """
    Compute the difference between two rotation matrices.
    Args:
        rotmat_ref: reference rotation matrix 1, of size (N, 3, 3)
        rotmat_target: target rotation matrix 2, of size (N, 3, 3)
        squared: if True (default), use the squared norm,
                otherwise use the standard form.
        align_matrix: alignment matrix for transforming 
            rotmat_target to rotmat_ref, of size (3, 3)
        ord: order of the norm (default: "fro")
           --------------------------------------
            ord     matrix norm
           --------------------------------------
            fro     Frobenius norm
            nuc     nuclear norm
            inf     max(sum(abs(x), dim=1))
           -inf     min(sum(abs(x), dim=1))
            1       max(sum(abs(x), dim=0))
           -1       min(sum(abs(x), dim=0))
            2       largest singular value
           -2       smallest singular value
           --------------------------------------
           https://pytorch.org/docs/stable/generated/torch.linalg.matrix_norm.html

    Returns:
        rotation matrix error (N,)
    """
    N = rotmat_ref.size(dim=0)  # number of samples

    if align_matrix is not None:
        rotmat_1 = align_matrix.expand(N, 3, 3) @ rotmat_target
    else:
        rotmat_1 = rotmat_target

    diff = rotmat_ref - rotmat_1
    norm = torch.linalg.matrix_norm(diff, ord=ord, dim=(-2, -1))
    if squared:
        return torch.pow(norm, 2)
    else:
        return norm
