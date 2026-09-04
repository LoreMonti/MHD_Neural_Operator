"""Linear growth-rate measurement and the theory we validate against.

During the early ("linear") phase of an instability the perturbation grows like
exp(gamma * t), so an energy (quadratic in the perturbation) grows like exp(2*gamma*t).
Taking the logarithm turns that into a straight line whose slope is 2*gamma:

    ln E(t) = const + 2*gamma*t.

`growth_rate` fits that line over the clean exponential window and returns gamma.

Theory references
-----------------
- Hydrodynamic tanh shear layer U0*tanh(y/delta): the fastest-growing mode has
  wavenumber k*delta ~= 0.4446 and temporal growth rate gamma ~= 0.1897 * U0/delta
  (Michalke 1964).
- Aligned magnetic field: a flow-parallel field of Alfven speed v_A stabilizes KH once
  it is strong enough. For a vortex sheet the marginal condition is delta_u = 2*v_A,
  i.e. the instability survives only for M_A = delta_u/v_A > 2 (Chandrasekhar 1961).
"""

from __future__ import annotations

import torch


def growth_rate(
    times: torch.Tensor,
    energy: torch.Tensor,
    window: tuple[float, float] | None = None,
    lo_frac: float = 1e2,
    hi_frac: float = 0.1,
) -> dict:
    """Fit gamma from an energy time series assuming energy ~ exp(2*gamma*t).

    Parameters
    ----------
    times, energy : 1D tensors of equal length.
    window : (t_min, t_max), optional
        Explicit fitting window. If omitted, the window is auto-selected as the band
        where the energy is between `lo_frac * energy_min` and `hi_frac * energy_max`
        (i.e. above the initial transient and below saturation).

    Returns
    -------
    dict with `gamma`, `r2` (fit quality), and the fit `window` actually used.
    """
    t = times.double()
    y = torch.log(energy.double().clamp_min(1e-300))

    if window is not None:
        mask = (t >= window[0]) & (t <= window[1])
    else:
        e = energy.double()
        lo = lo_frac * e.min()
        hi = hi_frac * e.max()
        mask = (e >= lo) & (e <= hi)
        if mask.sum() < 3:  # fall back to the middle 60% of the series
            n = len(t)
            mask = torch.zeros(n, dtype=torch.bool)
            mask[int(0.2 * n): int(0.8 * n)] = True

    tt, yy = t[mask], y[mask]
    if len(tt) < 2:
        raise ValueError("Not enough points in the fitting window to estimate a slope.")

    # Ordinary least squares slope/intercept.
    tbar, ybar = tt.mean(), yy.mean()
    slope = ((tt - tbar) * (yy - ybar)).sum() / ((tt - tbar) ** 2).sum()
    intercept = ybar - slope * tbar
    resid = yy - (intercept + slope * tt)
    ss_res = (resid**2).sum()
    ss_tot = ((yy - ybar) ** 2).sum().clamp_min(1e-300)
    r2 = float(1.0 - ss_res / ss_tot)

    return {
        "gamma": float(slope) / 2.0,   # energy grows as exp(2*gamma*t)
        "r2": r2,
        "window": (float(tt.min()), float(tt.max())),
    }


def michalke_max_growth(U0: float, delta: float) -> float:
    """Fastest-growing hydrodynamic KH rate for U0*tanh(y/delta): ~0.1897 * U0/delta."""
    return 0.1897 * U0 / delta


def michalke_max_wavenumber(delta: float) -> float:
    """Wavenumber of the fastest-growing mode: k*delta ~= 0.4446."""
    return 0.4446 / delta


# Marginal Alfvenic Mach number for an aligned field (vortex-sheet result).
M_A_CRITICAL = 2.0
