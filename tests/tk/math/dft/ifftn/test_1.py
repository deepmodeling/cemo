import torch
import itertools
import pytest
from cemo.tk.math.dft import fftn, ifftn
Tensor = torch.Tensor


# Test predefined input matrices
@pytest.mark.parametrize(
    (
        "x",
        "symm_freq",
        "symm_real",
        "use_rfft",
    ),
    itertools.product(
        [
            torch.tensor([0., 1., 2., 3., 4.]),
            torch.tensor([[0., 1., 2.], [3., 4., 5.], [6., 7., 8.]]),
            torch.tensor([[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]])
        ],
        [False, True],  # symm_freq
        [False, True],  # symm_real
        [False, True],  # use_rfft
    )
)
def test(
        x: Tensor,
        symm_freq: bool,
        symm_real: bool,
        use_rfft: bool,
        ):
    print("===================")
    print("test ifftn")
    print(f"{x=}")
    print(f"{symm_freq=}")
    print(f"{symm_real=}")
    print(f"{use_rfft=}")
    print("===================")
    dims = tuple(range(x.ndim))
    s = x.shape
    if symm_freq and use_rfft:
        with pytest.raises(ValueError):
            fftn(
                x,
                dims=dims,
                symm_freq=symm_freq,
                symm_real=symm_real,
                use_rfft=use_rfft)
        return

    x_ft = fftn(
        x,
        dims=dims,
        symm_freq=symm_freq,
        symm_real=symm_real,
        use_rfft=use_rfft)

    result = ifftn(
        x_ft,
        dims=dims,
        symm_freq=symm_freq,
        symm_real=symm_real,
        use_rfft=use_rfft,
        s=s).real

    assert torch.allclose(x, result, atol=1e-6)
