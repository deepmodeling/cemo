import torch
from cemo.tk.math.dft.perturb_coeffs import perturb_coeffs


def test_perturb_coeffs():
    inplace = False
    # Create a tensor
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    # Create an index tensor
    i_rows = torch.tensor([0, 1])
    i_cols = torch.tensor([1, 1])
    idx = [i_rows, i_cols]

    # Call the function
    result = perturb_coeffs(x, idx, inplace)

    # Check the values of the result
    assert result[0, 0] == 1.0
    assert result[0, 1] != 2.0
    assert result[1, 0] == 3.0
    assert result[1, 1] != 4.0


def test_perturb_coeffs_with_zeros():
    inplace = False
    # Create a tensor
    x = torch.zeros((2, 2))

    # Create an index tensor
    i_rows = torch.tensor([0, 1])
    i_cols = torch.tensor([1, 1])
    idx = [i_rows, i_cols]

    print(f"x \n{x}")
    print(f"x[idx] {x[idx]}")
    print(x[idx].shape)

    # Call the function
    result = perturb_coeffs(x, idx, inplace)

    print(f"result shape: {result.shape}")
    # Check the values of the result
    assert result[0, 0] == 0.0
    assert result[0, 1] != 0.0
    assert result[1, 0] == 0.0
    assert result[1, 1] != 0.0
