"""Kinetic energy spectrum: how energy is distributed across spatial scales.

A faithful surrogate must reproduce not just the fields but their *spectrum* — the
balance between large vortices (low k) and fine structures (high k). We compute the
shell-averaged kinetic energy E(k) from the vorticity field.
"""

from __future__ import annotations

import numpy as np
import torch

from ..solver.spectral import SpectralGrid


def kinetic_energy_spectrum(omega: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Shell-averaged kinetic energy spectrum E(k) from a vorticity field (H, W).

    Returns (k, E) where k are integer wavenumber shells.
    """
    n = omega.shape[-1]
    grid = SpectralGrid(n=n, dtype=torch.float64)
    psi_hat = grid.psi_hat_from_omega_hat(grid.fft(omega.double()))
    v_x, v_y = grid.velocity_from_psi_hat(psi_hat)
    vxh = grid.fft(v_x); vyh = grid.fft(v_y)
    e = 0.5 * (vxh.abs() ** 2 + vyh.abs() ** 2) / (n ** 4)   # spectral energy density

    kmag = torch.sqrt(grid.kx ** 2 + grid.ky ** 2).numpy()
    e = e.numpy()
    kbin = np.round(kmag).astype(int)
    kmax = n // 2
    ks = np.arange(1, kmax)
    spec = np.array([e[kbin == k].sum() for k in ks])
    return ks, spec
