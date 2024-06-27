from dataclasses import dataclass
import torch
Tensor = torch.Tensor


@dataclass
class CTF_Params:
    """
    a dataclass for CTF parameters.
    """
    df1_A: Tensor
    df2_A: Tensor
    df_angle_rad: Tensor
    accel_kv: Tensor
    cs_mm: Tensor
    amp_contrast: Tensor
    phase_shift_rad: Tensor
    bfactor: Tensor

    def __getitem__(self, idx: int) -> dict:
        return dict(
            df1_A=self.df1_A[idx],
            df2_A=self.df2_A[idx],
            df_angle_rad=self.df_angle_rad[idx],
            accel_kv=self.accel_kv[idx],
            cs_mm=self.cs_mm[idx],
            amp_contrast=self.amp_contrast[idx],
            phase_shift_rad=self.phase_shift_rad[idx],
            bfactor=self.bfactor[idx],
        )
