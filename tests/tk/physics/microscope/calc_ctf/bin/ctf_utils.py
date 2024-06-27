import torch
import numpy as np
import matplotlib.pyplot as plt
from cemo.tk.physics.microscope import CTF_Params


def plot_ctf(f_png: str, ctf: np.ndarray):
    plt.matshow(ctf)
    # sns.heatmap(ctf)
    plt.savefig(f_png)


def mk_ctf_params(
        batch_size: int = 0,
        defocusU: float = 15000.,
        defocusV: float = 16000.,
        defocusAngle: float = 45.,
        kV: float = 300.,
        cs: float = 0.01,
        w: float = 0.1,
        phase_shift: float = 0.,
        bfactor: float = 0.,
        ) -> CTF_Params:
    def to_tensor(x):
        output = torch.tensor([x], dtype=torch.float32)
        if batch_size <= 1:
            return output
        else:
            return output.expand(batch_size)

    return CTF_Params(
        df1_A=to_tensor(defocusU),
        df2_A=to_tensor(defocusV),
        df_angle_rad=torch.deg2rad(to_tensor(defocusAngle)),
        accel_kv=to_tensor(kV),
        cs_mm=to_tensor(cs),
        amp_contrast=to_tensor(w),
        phase_shift_rad=torch.deg2rad(to_tensor(phase_shift)),
        bfactor=to_tensor(bfactor),
    )


def mk_2d_freqs(
        L: int,
        apix: float,
        dtype: torch.dtype = torch.float64,
        ) -> torch.Tensor:
    ax = torch.linspace(-0.5, 0.5, L+1, dtype=dtype)[:-1]
    xs, ys = torch.meshgrid([ax, ax], indexing="ij")
    # cryoSPARC's 2D grid is in (Y, X) order
    freq2d = torch.stack([ys, xs], axis=-1) / apix
    return freq2d
