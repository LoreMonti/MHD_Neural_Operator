"""Kelvin-Helmholtz initial conditions.

We build the starting state of the plasma: two layers sliding past each other (the
shear flow) threaded by a magnetic field aligned with the flow, plus a tiny random
nudge that seeds the instability.

State representation
--------------------
The solver evolves two periodic scalar fields:
    omega   vorticity of the flow, omega = dv_y/dx - dv_x/dz  (here the curl's z-comp.)
    a       the *fluctuating* part of the magnetic flux potential

The aligned mean field is kept separate as a constant `B0` (a parameter, not a field),
because a uniform field B = (B0, 0) has flux potential A = B0 * y, which is **not
periodic** and cannot live on a Fourier grid. So we split

    A(x, y) = B0 * y + a(x, y),     a periodic,

giving the physical magnetic field

    B = curl(A z_hat) = (B0 + da/dy, -da/dx).

At t = 0 the field is uniform, so a = 0 and B = (B0, 0). This also matches the plan of
*conditioning the FNO on B0* (equivalently on the Alfvenic Mach number M_A): the mean
field is an input parameter, the fluctuation `a` is a learned field.

Units
-----
We use normalized units mu0 = 1 and rho = 1 by default, so the Alfven speed is
v_A = B0 / sqrt(mu0 * rho) = B0, and M_A = delta_u / v_A gives B0 = delta_u / M_A.
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import torch

from .spectral import SpectralGrid


@dataclass
class KHState:
    """Initial state produced for a Kelvin-Helmholtz run."""

    omega: torch.Tensor   # vorticity field, shape (N, N)
    a: torch.Tensor       # fluctuating flux potential, shape (N, N) (zeros at t=0)
    B0: float             # mean (aligned) field strength
    params: dict          # bookkeeping: M_A, delta_u, Re, etc.


def _coordinates(grid: SpectralGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Real-space (x, y) coordinate fields on the periodic grid, in float64."""
    n, length = grid.n, grid.length
    coord = torch.arange(n, dtype=grid.dtype, device=grid.device) * (length / n)
    xx = coord.view(n, 1).expand(n, n)
    yy = coord.view(1, n).expand(n, n)
    return xx, yy


def kelvin_helmholtz_state(
    grid: SpectralGrid,
    M_A: float,
    delta_u: float = 1.0,
    shear_thickness: float | None = None,
    perturbation_amp: float = 1.0e-3,
    n_modes: int = 4,
    seed: int = 0,
    rho: float = 1.0,
    mu0: float = 1.0,
) -> KHState:
    """Construct a Kelvin-Helmholtz initial state.

    Parameters
    ----------
    grid : SpectralGrid
        The spectral grid the fields live on.
    M_A : float
        Alfvenic Mach number, delta_u / v_A. Sets the mean field: B0 = delta_u / M_A
        (in normalized units). This is the key stabilization parameter.
    delta_u : float
        Total velocity jump across the shear layers.
    shear_thickness : float, optional
        Width of the tanh shear layer. Defaults to 0.05 * box length (a few grid cells).
    perturbation_amp : float
        Amplitude of the seed velocity perturbation, as a fraction of delta_u.
    n_modes : int
        Number of low-wavenumber x-modes in the seed perturbation.
    seed : int
        RNG seed for reproducible perturbations.
    rho, mu0 : float
        Density and vacuum permeability (normalized units by default).
    """
    n, length = grid.n, grid.length
    if shear_thickness is None:
        shear_thickness = 0.05 * length
    xx, yy = _coordinates(grid)

    # --- base shear flow: a periodic double shear layer -----------------------------
    # v_x jumps between +delta_u/2 and -delta_u/2 across two interfaces (at L/4, 3L/4),
    # so the profile is periodic in y (required on a Fourier grid).
    y1, y2 = 0.25 * length, 0.75 * length
    a_th = shear_thickness
    v_x = 0.5 * delta_u * (
        torch.tanh((yy - y1) / a_th) - torch.tanh((yy - y2) / a_th) - 1.0
    )

    # --- seed perturbation in v_y, localized near the two interfaces -----------------
    gen = torch.Generator(device="cpu").manual_seed(seed)
    envelope = torch.exp(-((yy - y1) / a_th) ** 2) + torch.exp(-((yy - y2) / a_th) ** 2)
    v_y = torch.zeros_like(v_x)
    for m in range(1, n_modes + 1):
        amp_m = torch.randn(1, generator=gen).item()
        phase_m = 2 * math.pi * torch.rand(1, generator=gen).item()
        v_y = v_y + amp_m * torch.sin(2 * math.pi * m * xx / length + phase_m)
    v_y = v_y * envelope
    # Normalize so the peak perturbation is perturbation_amp * delta_u.
    peak = v_y.abs().max()
    if peak > 0:
        v_y = v_y * (perturbation_amp * delta_u / peak)

    # --- vorticity from the velocity field (spectral curl) --------------------------
    # omega = dv_y/dx - dv_x/dy
    vx_hat, vy_hat = grid.fft(v_x), grid.fft(v_y)
    dvy_dx_hat, _ = grid.grad(vy_hat)
    _, dvx_dy_hat = grid.grad(vx_hat)
    omega = grid.ifft(dvy_dx_hat - dvx_dy_hat)

    # --- magnetic field: uniform aligned mean field, no fluctuation yet --------------
    v_A = B0_from_M_A(M_A, delta_u=delta_u, rho=rho, mu0=mu0)
    B0 = v_A  # in normalized units v_A = B0
    a = torch.zeros_like(omega)

    params = {
        "M_A": M_A,
        "delta_u": delta_u,
        "shear_thickness": shear_thickness,
        "perturbation_amp": perturbation_amp,
        "n_modes": n_modes,
        "seed": seed,
        "rho": rho,
        "mu0": mu0,
        "B0": B0,
    }
    return KHState(omega=omega, a=a, B0=B0, params=params)


def B0_from_M_A(M_A: float, delta_u: float = 1.0, rho: float = 1.0, mu0: float = 1.0) -> float:
    """Mean field strength for a target Alfvenic Mach number.

    M_A = delta_u / v_A with v_A = B0 / sqrt(mu0 * rho)  =>  B0 = delta_u / M_A * sqrt(mu0*rho).
    """
    v_A = delta_u / M_A
    return v_A * math.sqrt(mu0 * rho)
