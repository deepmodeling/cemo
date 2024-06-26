import torch
Tensor = torch.Tensor


def mm2angs(x: Tensor) -> Tensor:
    """Convert unit from mm to angstrom"""
    factor = 10 ** 7
    return x * factor
