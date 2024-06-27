import torch
from torch import Tensor
from typing import Optional, Union
import numpy
Float_or_Tensor = Union[float, Tensor]


def deg2rad(x: Float_or_Tensor) -> Float_or_Tensor:
    return x * torch.pi / 180.


def mm2angs(x: Float_or_Tensor) -> Float_or_Tensor:
    """Convert unit from mm to angstrom"""
    factor = 10 ** 7
    return x * factor


def sqrt(x: Float_or_Tensor) -> Float_or_Tensor:
    return x ** 0.5


def square(x: Float_or_Tensor) -> Float_or_Tensor:
    return x ** 2


def atan2(y: Float_or_Tensor, x: Float_or_Tensor) -> Float_or_Tensor:
    assert type(y) == type(x)

    if type(y) is Tensor and type(x) is Tensor:
        return torch.arctan2(y, x)
    else:
        return numpy.arctan2(y, x)


def calc_ctf(
        freq_grid: Tensor,
        z_max: Float_or_Tensor = 0.,
        z_min: Float_or_Tensor = 0.,
        theta: Float_or_Tensor = 0.,
        kV: Float_or_Tensor = 300.,
        Cs: Float_or_Tensor = 0.,
        w: Float_or_Tensor = 1.0,
        phase_shift: Float_or_Tensor = 0.,
        b_factor: Optional[Float_or_Tensor] = None,
        ) -> Tensor:
    """
    Calculate CTF for 2D EM images.
    Reference:
    Marabini, et. al. "The Electron Microscopy eXchange (EMX) initiative",
    Journal of Structural Biology, 194: 156-163 (2016).

    Args:
        freq_grid: 2D frequency grids of shape (L, L, 2) and (B, L, L, 2)
            or (N, 2) and (B, N, 2)
            where B is the batch size and L is the length of the input image 
            (one edge) and N = L*L
        z_max: max defocus (a.k.a. defocusU) (unit: angstrom).
            shape: float or (B, 1)
        z_min: min defocus (a.k.a. defocusV) (unit: angstrom).
            shape: float or (B, 1)
        theta: astigmatism angle (a.k.a. defocusAngle, unit: degrees),
            i.e., angle between the maximum defocus direction and the X axis.
            shape: float or (B, 1)
        kV: microsope voltage (unit: kV)
            shape: float or (B, 1)
        Cs: spherical aberration (unit: mm)
            shape: float or (B, 1)
        w: fraction of amplitude contrast (between 0 and 1)
            shape: float or (B, 1)
        phase_shift: phase shift (unit: degrees)
            shape: float or (B, 1)
        b_factor: B-factor (Unit: angstrom^2)
            shape: float or (B, 1)
    Returns:
        a 2D tensor (L, L) or a stack of 2D tensors (B, L, L)
    """
    assert freq_grid.shape[-1] == 2

    Cs_A = mm2angs(Cs)  # unit: angstrom
    theta_rad = deg2rad(theta)
    phase_shift_rad = deg2rad(phase_shift)

    # electron related parameter
    Lambda = 12.2639 / sqrt(1000 * kV + 0.97845 * square(kV))
    Lambda2 = square(Lambda)

    xs = freq_grid[..., 0]
    ys = freq_grid[..., 1]
    ang = atan2(ys, xs)
    S2 = square(xs) + square(ys)
    S4 = square(S2)

    dZ_avg = 0.5 * (z_max + z_min)
    dZ_diff = 0.5 * (z_max - z_min)
    dZ = dZ_avg + dZ_diff * torch.cos(2. * (ang - theta_rad))

    g0 = torch.pi * Lambda
    g1 = -dZ * S2
    g2 = 0.5 * Cs_A * S4 * Lambda2
    gamma = g0 * (g1 + g2) - phase_shift_rad

    w2 = square(w)
    ctf = -w * torch.cos(gamma) + sqrt(1. - w2) * torch.sin(gamma)

    if b_factor is None:
        return ctf
    else:
        return ctf * torch.exp(-S2 * (b_factor / 4.))
