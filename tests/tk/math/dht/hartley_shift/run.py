import torch
import os
from cemo.tk.math import dht
from cemo.tk import index, plot
from typing import List
Tensor = torch.Tensor


def run(
        output_dir: str,
        shape: List[int],
        dims: List[int],
        shift: Tensor,
        indexing: str = "ij",
        symm: bool = False,
        debug: bool = True):
    """
    Test the fourier_shift function.
    """
    print()
    print("="*60)
    print(f"output_dir: {output_dir}")
    print(f"shape: {shape}")
    print(f"dims: {dims}")
    print(f"shift: {shift}")
    print(f"indexing: {indexing}")
    print(f"symm: {symm}")
    print("="*60)
    os.makedirs(output_dir, exist_ok=True)
    x = torch.rand(shape)

    x_ht = dht.htn(x, dims=dims, symm=symm)

    y_ft = dht.hartley_shift(
        x=x_ht,
        shift=shift,
        dims=dims,
        indexing=indexing,
        symm=symm,
        debug=debug,
    )

    y = dht.ihtn(y_ft, dims=dims, symm=symm).real

    y_expect = index.shift2d(x, shift, indexing=indexing)

    assert torch.allclose(y, y_expect, atol=1e-6, rtol=0), \
        f"Matrices are not equal:\nx=\n{x}\ny=\n{y}\ny_expect=\n{y_expect}"

    # fig, _ = plot.plot_mats(
    #     [x, y_expect, y], ["x", "y_expect", "y"])
    # fig.savefig(f"{output_dir}/test_1.png")
