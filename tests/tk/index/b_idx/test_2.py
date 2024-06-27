import pytest
import torch
import copy
from cemo.tk.index import b_idx
Tensor = torch.Tensor


@pytest.mark.parametrize(
    (
        "input",
        "expect"
    ),
    [
        (
            [
                torch.tensor([1, 2, 3]),
                torch.tensor([4, 5, 6]),
                torch.tensor([7, 8, 9]),
            ],
            [
                torch.tensor([1, 2, 3]).view([-1, 1, 1]),
                torch.tensor([4, 5, 6]).view([1, -1, 1]),
                torch.tensor([7, 8, 9]).view([1, 1, -1]),
            ]
        ),
        (
            [slice(1, 4), slice(4, 7), slice(7, 10)],
            [slice(1, 4), slice(4, 7), slice(7, 10)]
        ),
        (
            [
                torch.tensor([1, 2, 3]),
                slice(4, 7),
                torch.tensor([7, 8, 9]),
            ],
            [
                torch.tensor([1, 2, 3]).view([-1, 1]),
                slice(4, 7),
                torch.tensor([7, 8, 9]).view([1, -1]),
            ],
        ),
    ]
)
def test_b_idx_with_multiple_inputs(
    input,
    expect):
    print("==================")
    print(f"input =\n{input}")
    print(f"expect =\n{expect}")
    result = b_idx(input)
    print(f"result =\n{result}")
    assert len(result) == len(expect)
    for r, e in zip(result, expect):
        if isinstance(r, torch.Tensor):
            assert torch.equal(r, e)
        else:
            assert r == e
    
    N = 10
    x = torch.rand(N, N, N)

    tensor_dims = list(
        filter(lambda i: type(input[i]) == Tensor, range(len(input)))
    )
    expect2 = copy.deepcopy(input)

    tensor_ids = list(map(lambda i: input[i], tensor_dims))
    if len(tensor_ids) == 0:
        id_list = []
    else:
        id_list = torch.meshgrid(tensor_ids, indexing="ij")

    def update(i: int):
        d = tensor_dims[i]
        expect2[d] = id_list[i]

    _ = list(map(update, range(len(tensor_ids))))

    print(f"expect2 =\n{expect2}")
    torch.testing.assert_close(
        x[result],
        x[expect2],
    )

    print("==================")