import torch
from test_tk import cs_file_name
from cemo.io import cs
from cemo.io.cs.get_shift import get_shift


def test_get_shift():
    cs_fid = 1
    cs_obj = cs.read(cs_file_name(cs_fid))
    dtype = torch.float32
    is_abinit = False
    return_ratio = True
    verbose = True
    N = len(cs_obj.data)

    shift = get_shift(
        cs_obj,
        is_abinit=is_abinit,
        return_ratio=return_ratio,
        dtype=dtype,
        verbose=verbose,
    )

    shift_px = cs_obj.data["alignments3D/shift"]
    shape = cs_obj.data["blob/shape"].astype(dtype=shift_px.dtype)
    shift_expect = torch.tensor(shift_px/shape, dtype=dtype)

    assert torch.is_tensor(shift)
    assert shift.shape == (N, 2)
    assert torch.allclose(shift, shift_expect)
