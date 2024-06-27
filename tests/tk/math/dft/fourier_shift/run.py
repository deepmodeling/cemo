import torch
import os
import pytest
from cemo.tk.math import dft
from cemo.tk import index
from typing import List
Tensor = torch.Tensor


def run(
        output_dir: str,
        shape: List[int],
        dims: List[int],
        d: float,
        shift: Tensor,
        indexing: str = "ij",
        is_rfft: bool = False,
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
    print(f"d: {d}")
    print(f"shift: {shift}")
    print(f"indexing: {indexing}")
    print(f"is_rfft: {is_rfft}")
    print(f"symm: {symm}")
    print("="*60)
    os.makedirs(output_dir, exist_ok=True)
    x = torch.rand(shape)

    if is_rfft:
        shape_ft = tuple(x.shape[i] for i in dims)
        x_ft = torch.fft.rfftn(x, dim=dims)
    else:
        shape_ft = None
        x_ft = dft.fftn(x, dims=dims, symm=symm)

    if is_rfft and symm:
        with pytest.raises(ValueError):
            y_ft = dft.fourier_shift(
                x=x_ft,
                s=shape_ft,
                shift=shift,
                dims=dims,
                d=d,
                indexing=indexing,
                is_rfft=is_rfft,
                symm=symm,
                debug=debug,
            )
        return
    else:
        y_ft = dft.fourier_shift(
            x=x_ft,
            s=shape_ft,
            shift=shift,
            dims=dims,
            d=d,
            indexing=indexing,
            is_rfft=is_rfft,
            symm=symm,
            debug=debug,
        )

    if is_rfft:
        y = torch.fft.irfftn(y_ft, s=shape_ft, dim=dims).real
    else:
        y = dft.ifftn(y_ft, dims=dims, symm=symm).real

    y_expect = index.shift2d(x, shift * d, indexing=indexing)

    assert torch.allclose(y, y_expect, atol=1e-6, rtol=0), \
        f"Matrices are not equal:\nx=\n{x}\ny=\n{y}\ny_expect=\n{y_expect}"

    # fig, _ = plot.plot_mats(
    #     [x, y_expect, y], ["x", "y_expect", "y"])
    # fig.savefig(f"{output_dir}/test_1.png")
