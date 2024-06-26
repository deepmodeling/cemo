import torch
Tensor = torch.Tensor


def assert_mat_eq(x: Tensor, y: Tensor, tol: float = 1e-7):
    assert torch.allclose(x, y, atol=tol), \
        f"Matrices are not equal:\n{x}\n{y}"
