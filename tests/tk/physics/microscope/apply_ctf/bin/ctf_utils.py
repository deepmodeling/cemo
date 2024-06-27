import torch
import numpy as np
import matplotlib.pyplot as plt
from cemo.tk.physics.microscope import CTF_Params


def plot_ctf(f_png: str, ctf: np.ndarray):
    plt.matshow(ctf)
    # sns.heatmap(ctf)
    plt.savefig(f_png)


def mk_ctf_params(
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
        return torch.tensor([x], dtype=torch.float32)

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


def mk_2d_freqs(L: int, apix: float) -> torch.Tensor:
    ax = np.linspace(-0.5, 0.5, L, endpoint=False)
    # ax = np.linspace(-0.5, 0.5, L, endpoint=True)
    xs, ys = np.meshgrid(ax, ax, indexing="xy")
    freq2d = torch.tensor(np.stack([xs, ys], axis=-1) / apix)
    return freq2d
