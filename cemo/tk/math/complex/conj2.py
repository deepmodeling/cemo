import torch
Tensor = torch.Tensor


def conj2(x: Tensor) -> Tensor:
    """
    This is a BFloat16-compatible version of torch.conj.
    Complex conjugate of a complex tensor expressed as separate values
    for the real and imaginary parts.
    (the size of the last dimension must be 2)

    Args:
        x: a complex tensor
        dim: the dimension along which to take the conjugate

    Returns:
        A complex tensor
    """
    assert x.shape[-1] == 2
    return torch.stack(
        (x[..., 0], -x[..., 1]),
        dim=-1,
    )
