import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_ctf(f_png: str, ctf: np.ndarray):
    plt.matshow(ctf)
    # sns.heatmap(ctf)
    plt.savefig(f_png)


def mk_ctf_params(
        defocusU: float = 15000.,
        defocusV: float = 16000.,
        defocusAngle: float = 80.,
        kV: float = 300.,
        cs: float = 0.01,
        w: float = 0.1,
        phase_shift: float = 0.) -> np.ndarray:

    return np.array(
        [
            defocusU,
            defocusV,
            defocusAngle,
            kV,
            cs,
            w,
            phase_shift,
        ]
    )


def mk_2d_freqs(L: int, apix: float) -> torch.Tensor:
    ax = np.linspace(-0.5, 0.5, L, endpoint=False)
    # ax = np.linspace(-0.5, 0.5, L, endpoint=True)
    xs, ys = np.meshgrid(ax, ax, indexing="xy")
    freq2d = torch.tensor(np.stack([xs, ys], axis=-1) / apix)
    return freq2d
