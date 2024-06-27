import torch
from cemo.tk.physics.microscope.CTF_Params import CTF_Params 


def test_valid_index():
    # Should return a CTF_Params object with the correct values when given a valid index.
    params = CTF_Params(
        df1_A=torch.tensor([1, 2, 3]),
        df2_A=torch.tensor([4, 5, 6]),
        df_angle_rad=torch.tensor([7, 8, 9]),
        accel_kv=torch.tensor([10, 11, 12]),
        cs_mm=torch.tensor([13, 14, 15]),
        amp_contrast=torch.tensor([16, 17, 18]),
        phase_shift_rad=torch.tensor([19, 20, 21]),
        bfactor=torch.tensor([22, 23, 24]),
    )
    result = params[1]
    assert result.df1_A == 2
    assert result.df2_A == 5
    assert result.df_angle_rad == 8
    assert result.accel_kv == 11
    assert result.cs_mm == 14
    assert result.amp_contrast == 17
    assert result.phase_shift_rad == 20
    assert result.bfactor == 23
