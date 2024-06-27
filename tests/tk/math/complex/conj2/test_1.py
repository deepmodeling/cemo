import torch
from cemo.tk.math.complex import conj2


def test_conj2():
    # Create a complex tensor
    batch_size = 10
    N = 5
    shape = (batch_size, N, N, 2)
    dtype = torch.bfloat16
    x = torch.rand(shape, dtype=dtype)

    # Compute the complex conjugate using the conj2 function
    result = conj2(x)

    # Expected result
    expected = torch.stack(
        (x[..., 0],
        -x[..., 1]),
        dim=-1,
    )

    # Check if the result is close to the expected result
    assert torch.allclose(result, expected, atol=1e-6, rtol=0.)
