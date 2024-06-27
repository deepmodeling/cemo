import torch
from typing import Optional
from dataclasses import fields, Field
from cemo.tk.physics.microscope.CTF_Params import CTF_Params
from cemo.tk.unitconvert import mm2angs
Tensor = torch.Tensor


# self, freqs, dfu, dfv, dfang, volt, cs, w, phase_shift=0, bfactor=None
def calc_ctf(
        freqs: Tensor,
        params: Optional[CTF_Params],
        apix: Optional[float] = None,
        ) -> Tensor:
    """
    Compute the Contrast Transfer Function (CTF)

    Args:
        freqs: 2D Fourier-space frequency grid of shape (L, 2) or (L, L, 2)
        params: CTF parameters. Each field is a tensor of shape (B, ).
        apix: pixel size in angstroms

    Returns:
        CTF filter matrix of shape (L,), (B, L), or (B, L, L)
        If params is None or params.accel_kv.shape[-1] == 0,
        returns a tensor filled with 1.
    """
    assert freqs.ndim in (2, 3), \
        f"freqs.ndim must be 2 or 3, but got {freqs.ndim}."
    assert freqs.shape[-1] == 2, \
        f"freqs.shape[-1] must be 2, but got {freqs.shape[-1]}."

    if apix is not None:
        freqs = freqs / apix

    dtype = freqs.dtype
    batch_size = len(params.df1_A) if params is not None else 0
    dim_filler = [1] * (freqs.ndim - 1)  # (1,) or (1, 1)
    if params is None or batch_size == 0:
        return torch.tensor([1.], dtype=dtype).expand(freqs.shape[:-1])
    elif batch_size == 1:
        params_new_shape = dim_filler  # (1,) or (1, 1)
    else:
        params_new_shape = (batch_size, *dim_filler)  # (B, 1) or (B, 1, 1)

    def reshape(k: Field):
        x = getattr(params, k.name)
        new_x = x.view(params_new_shape)
        setattr(params, k.name, new_x)

    # reshape params
    _ = list(map(reshape, fields(params)))

    NDim = freqs.shape[-1]
    assert (NDim == 2) or (NDim == 3), f"NDim must be 2 or 3, but got {NDim}."

    Cs = mm2angs(params.cs_mm)  # unit: Å
    theta = params.df_angle_rad  # unit: rad  (defocusUAngle)
    phase_shift = params.phase_shift_rad  # unit: rad
    kV = params.accel_kv  # unit: kV
    z_max = params.df1_A  # unit: Å （a.k.a. defocusU）
    z_min = params.df2_A  # unit: Å  (a.k.a. defocusV)
    bfactor = params.bfactor  # unit: Å^2
    w = params.amp_contrast  # unit: 1  (a.k.a. amplitude contrast)

    # electron related parameter
    Lambda = 12.2639 / torch.sqrt(1000 * kV + 0.97845 * kV.pow(2))  # Å
    Lambda2 = Lambda ** 2  # Å^2

    xs = freqs[..., 0]  # 1/Å; shape (L,) or (L, L)
    ys = freqs[..., 1]  # 1/Å; shape (L,) or (L, L)
    ang = torch.atan2(ys, xs)  # rad; shape (L,) or (L, L)
    S2 = xs.pow(2) + ys.pow(2)  # 1/Å^2; shape (L,) or (L, L)
    S4 = S2.pow(2)  # 1/Å^4; shape (L,) or (L, L)

    dZ_avg = 0.5 * (z_max + z_min)
    dZ_diff = 0.5 * (z_max - z_min)
    dZ = dZ_avg + dZ_diff * torch.cos(2. * (ang - theta))

    g0 = torch.pi * Lambda
    g1 = -dZ * S2
    g2 = 0.5 * Cs * S4 * Lambda2
    gamma = g0 * (g1 + g2) - phase_shift

    w2 = w.pow(2)
    ctf = -w * torch.cos(gamma) + torch.sqrt(1. - w2) * torch.sin(gamma)

    bfactor_is_zero = torch.allclose(bfactor, torch.zeros_like(bfactor))
    if not bfactor_is_zero:
        return ctf
    else:
        return ctf * torch.exp(-S2 * (bfactor / 4.))
