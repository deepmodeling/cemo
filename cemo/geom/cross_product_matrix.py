import torch


def cross_product_matrix(k: torch.Tensor) -> torch.Tensor:
    """
    Define the cross-product matrix K.
    K = [0, -kz, ky; 
         kz, 0, -kx; 
        -ky, kx, 0]
    https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    """
    kx = k[..., 0]
    ky = k[..., 1]
    kz = k[..., 2]
    K = torch.zeros(k.shape[:-1] + (3, 3), device=k.device, dtype=k.dtype)
    K[..., 0, 1] = -kz
    K[..., 1, 0] = kz
    K[..., 0, 2] = ky
    K[..., 2, 0] = -ky
    K[..., 1, 2] = -kx
    K[..., 2, 1] = kx
    return K
