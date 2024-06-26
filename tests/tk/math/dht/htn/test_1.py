import pytest
from cemo.tk.math import dht, dft
from cemo.tk.asserto import assert_mat_eq
import torch


@pytest.mark.parametrize("symm", [False, True])
def test(symm: bool):
    print(f"{symm=}")
    batch_size = 10
    image_size = 4
    shape = (batch_size, image_size, image_size)
    x = torch.rand(shape)
    dims = [-2, -1]
    y = dht.htn(x, dims=dims, symm_freq=symm, symm_real=True)

    x_ft = dft.fftn(x, dims=dims, symm_freq=symm, symm_real=True)
    y_expect = x_ft.real - x_ft.imag

    def aux(i: int):
        assert_mat_eq(y[i], y_expect[i])

    _ = list(map(aux, range(batch_size)))
