import torch
from scipy.spatial.transform import Rotation
from test_tk import cs_file_name
from cemo.io import cs
from cemo.io.cs.get_rotmat import get_rotmat


def test_get_rotmat():
    cs_fid = 1
    cs_obj = cs.read(cs_file_name(cs_fid))
    dtype = torch.float32
    is_abinit = False
    verbose = True
    N = len(cs_obj.data)

    rotmat = get_rotmat(
        cs_obj,
        is_abinit=is_abinit,
        dtype=dtype,
        verbose=verbose,
    )

    rotvec = cs_obj.data["alignments3D/pose"]
    rotmat_expect = torch.tensor(
        Rotation.from_rotvec(rotvec).as_matrix(),
        dtype=dtype)

    assert torch.is_tensor(rotmat)
    assert rotmat.shape == (N, 3, 3)
    assert torch.allclose(rotmat, rotmat_expect)
