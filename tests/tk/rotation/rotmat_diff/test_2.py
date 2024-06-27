from cemo.tk.rotation import rotmat_diff
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor


def print_mat(mat: Tensor):
    print(mat.numpy().round(2))


def test():
    X1 = torch.tensor(
        [
            [[1., 2, 3],
             [1., 1, 3],
             [4, 1, 1]],
        ],
        dtype=torch.float32)
    X2 = torch.tensor(
        [
            [[1, 2, 1],
             [3, 1, 4],
             [2, 5, 1]],
        ],
        dtype=torch.float32)
    R = torch.tensor(
        Rotation.from_euler("xyz", [90, 90, 90], degrees=True).as_matrix(),
        dtype=torch.float32,
    )
    diff = rotmat_diff(X1, X2, R, squared=True)
    expect = torch.linalg.matrix_norm(
            X1 - (R @ X2),
        )**2
    print("X1")
    print_mat(X1)
    print("X2")
    print_mat(X2)
    print("R")
    print_mat(R)
    print("R*X2")
    print_mat(R @ X2)
    print("diff")
    print_mat(diff)
    print("expect")
    print_mat(expect)
    assert torch.allclose(diff, expect)
