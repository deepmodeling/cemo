import torch
import numpy
from cemo.io import cs
from cemo.io.cs.get_ctf_params import get_ctf_params
Tensor = torch.Tensor
NumpyArray = numpy.ndarray


def test():
    f_input = "data/7eu7_P_res2.5_apix1_D256.cs"
    cs_obj = cs.read(f_input)
    ctf_params = get_ctf_params(cs_obj)
    keys_expect = (
        'df1_A',
        'df2_A',
        'df_angle_rad',
        'accel_kv',
        'cs_mm',
        'amp_contrast',
        'phase_shift_rad',
        'bfactor',
    )

    assert cs_obj.data.shape[0] == ctf_params.df1_A.shape[0]

    def to_tensor(x: NumpyArray) -> Tensor:
        return torch.tensor(
            x.copy(),
            dtype=torch.float32,
        )

    def check(k: str):
        src_k = f"ctf/{k}"
        src_data = to_tensor(cs_obj.data[src_k])
        new_data = getattr(ctf_params, k)
        print(src_k)
        torch.testing.assert_close(src_data, new_data)

    _ = list(map(check, keys_expect))
