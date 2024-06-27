import torch
from cemo.tk.math.dft import ifftn_symm, fftn_symm


# add batch dimension
def test():
    batch_size = 10
    x = torch.rand(batch_size, 3, 3)
    expected_output = x
    dims = (0, 1)
    y = ifftn_symm(fftn_symm(x, dims=dims), dims=dims).real

    def check(i: int):
        print(f"i = {i}")
        print(f"y[{i}] =\n{y[i]}")
        print(f"expected_output[{i}] =\n{expected_output[i]}")
        assert torch.allclose(y[i], expected_output[i], atol=1e-6, rtol=0.)

    _ = list(map(check, range(batch_size)))
