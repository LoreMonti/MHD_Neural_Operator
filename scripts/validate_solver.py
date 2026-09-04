"""Validate the pseudo-spectral MHD solver against linear theory.

Two checks:
  1. Hydrodynamic growth rate of a single seeded mode (compared to the Michalke
     tanh-layer estimate; note our double shear layer grows somewhat slower because
     the two interfaces interact at this wavelength).
  2. Magnetic stabilization threshold: gamma(M_A) must cross zero near M_A ~ 2
     (Chandrasekhar's aligned-field vortex-sheet result).

Usage:
    python scripts/validate_solver.py            # runs both checks, saves a figure
    python scripts/validate_solver.py --n 128    # grid resolution
"""

from __future__ import annotations

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mhd_fno.evaluation.growth import M_A_CRITICAL, growth_rate, michalke_max_growth
from mhd_fno.solver.initial import kelvin_helmholtz_state
from mhd_fno.solver.mhd2d import MHD2DSolver, transport_coeffs
from mhd_fno.solver.spectral import SpectralGrid


def hydro_check(n: int, delta: float, Re: float):
    g = SpectralGrid(n=n, dtype=torch.float64)
    nu, eta = transport_coeffs(Re=Re, delta_u=1.0, length=g.length)
    st = kelvin_helmholtz_state(
        g, M_A=1000.0, delta_u=1.0, shear_thickness=delta, single_mode=2, perturbation_amp=1e-3
    )
    solv = MHD2DSolver(g, nu=nu, eta=eta, B0=st.B0)
    out = solv.run(st, t_end=12.0, cfl=0.4, save_every=5)
    res = growth_rate(out["times"], out["E_y"])
    return out, res, michalke_max_growth(0.5, delta)


def threshold_scan(n: int, delta: float, Re: float, M_A_values):
    g = SpectralGrid(n=n, dtype=torch.float64)
    nu, eta = transport_coeffs(Re=Re, delta_u=1.0, length=g.length)
    gammas = []
    for M_A in M_A_values:
        st = kelvin_helmholtz_state(
            g, M_A=M_A, delta_u=1.0, shear_thickness=delta, single_mode=2, perturbation_amp=1e-3
        )
        solv = MHD2DSolver(g, nu=nu, eta=eta, B0=st.B0)
        out = solv.run(st, t_end=10.0, cfl=0.4, save_every=5)
        gammas.append(growth_rate(out["times"], out["E_y"], window=(1.5, 7.0))["gamma"])
    return torch.tensor(M_A_values), torch.tensor(gammas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--delta", type=float, default=0.222)
    parser.add_argument("--Re", type=float, default=5000.0)
    parser.add_argument("--out", default="notebooks/solver_validation.png")
    args = parser.parse_args()

    out, res, gamma_th = hydro_check(args.n, args.delta, args.Re)
    print("--- Hydrodynamic growth rate ---")
    print(f"measured gamma = {res['gamma']:.4f}  (R2={res['r2']:.4f})")
    print(f"Michalke single-layer estimate = {gamma_th:.4f}")

    M_A_values = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 8.0]
    M_A, gammas = threshold_scan(args.n, args.delta, args.Re, M_A_values)
    # linear-interpolated zero crossing of gamma(M_A)
    sign = gammas.sign()
    cross = ((sign[:-1] * sign[1:]) < 0).nonzero()
    if len(cross):
        i = int(cross[0])
        m0, m1, g0, g1 = M_A[i], M_A[i + 1], gammas[i], gammas[i + 1]
        M_A_thresh = float(m0 - g0 * (m1 - m0) / (g1 - g0))
    else:
        M_A_thresh = float("nan")
    print("\n--- Magnetic stabilization threshold ---")
    print(f"measured M_A threshold = {M_A_thresh:.2f}  (theory ~ {M_A_CRITICAL})")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].semilogy(out["times"], out["E_y"], "o-", ms=3)
    ax[0].set(xlabel="t", ylabel="E_y", title=f"Hydro growth (gamma={res['gamma']:.3f})")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axvline(M_A_CRITICAL, color="gray", ls="--", label="theory M_A=2")
    ax[1].plot(M_A, gammas, "o-")
    if not math.isnan(M_A_thresh):
        ax[1].axvline(M_A_thresh, color="C3", ls=":", label=f"measured {M_A_thresh:.2f}")
    ax[1].set(xlabel="M_A", ylabel="growth rate gamma", title="Magnetic stabilization threshold")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"\nfigure saved to {args.out}")


if __name__ == "__main__":
    main()
