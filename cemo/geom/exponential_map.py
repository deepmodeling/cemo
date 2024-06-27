import torch
from .cross_product_matrix import cross_product_matrix


def exponential_map(vec: torch.Tensor) -> torch.Tensor:
    """
    Exponetial map is a transformation that maps the
    axis-angle style rotation to a rotation matrix using
    the matrix form of Rodrigues' rotation formula.
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    In the axis-angle representation, the rotation angle
    is encoded as the length of vec, and the unit vector 
    of the axis of rotation has the same direction as vec.
    https://en.wikipedia.org/wiki/Axis-angle_representation 

    Args:
        vec: A tensor of shape (..., 3)
            representing the axis-angle style rotation.

    Returns:
        A tensor of shape (..., 3, 3) representing the rotation matrix.
    """
    angle = vec.norm(p=2, dim=-1, keepdim=True)
    length = angle

    # compute the unit vector of axis of rotation (i.e., k)
    k = vec / length

    # K: cross-product matrix 
    # (https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication)
    # Kv = k x v where k is the reference vector around which 
    # the rotation is defined. 
    K = cross_product_matrix(k)

    # identity matrix I
    Ieye = torch.eye(3, device=vec.device, dtype=vec.dtype)

    # rotation matrix R
    # R = I + sing(theta) * K + (1 - cos(theta)) * K^2
    # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    sin_a = torch.sin(angle).unsqueeze(dim=-1)
    cos_a = torch.cos(angle).unsqueeze(dim=-1)
    R = Ieye + sin_a * K + (1.0 - cos_a) * (K @ K)
    return R
