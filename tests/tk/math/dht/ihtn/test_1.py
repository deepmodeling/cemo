import pytest
import torch
import itertools
from cemo.tk.math import dht
from cemo.tk.asserto import assert_mat_eq


@pytest.mark.parametrize(
        (
            "symm_freq",
            "symm_real",
        ),
        itertools.product(
            [False, True],
            [False, True],
        ),
)
def test(symm_freq: bool, symm_real: bool):
    print("=========================================")
    print(f"{symm_freq=}")
    print(f"{symm_real=}")
    print("=========================================")
    batch_size = 5
    image_size = 4
    shape = (batch_size, image_size, image_size)
    x = torch.rand(shape)
    dims = [-2, -1]
    x_ht = dht.htn(x, dims=dims, symm_freq=symm_freq, symm_real=symm_real)
    y = dht.ihtn(x_ht, dims=dims, symm_freq=symm_freq, symm_real=symm_real)
    y_expect = x

    def aux(i: int):
        print(f"{i=}")
        # assert_mat_eq(y[i], y_expect[i])
        torch.testing.assert_close(y[i], y_expect[i])

    _ = list(map(aux, range(batch_size)))
