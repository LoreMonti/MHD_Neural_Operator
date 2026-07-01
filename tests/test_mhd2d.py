"""Tests for the 2D MHD solver.

These are cheap regression checks (small grid, short time). The quantitative
growth-rate-vs-linear-theory validation lives in the evaluation phase.
"""

import torch

from mhd_fno.solver.spectral import SpectralGrid
from mhd_fno.solver.initial import kelvin_helmholtz_state
from mhd_fno.solver.mhd2d import MHD2DSolver, transport_coeffs


def _run(M_A, n=64, t_end=8.0, seed=0):
    g = SpectralGrid(n=n, dtype=torch.float64)
    nu, eta = transport_coeffs(Re=2000, delta_u=1.0, length=g.length)
    st = kelvin_helmholtz_state(g, M_A=M_A, delta_u=1.0, perturbation_amp=1e-3, seed=seed)
    solv = MHD2DSolver(g, nu=nu, eta=eta, B0=st.B0)
    return solv.run(st, t_end=t_end, cfl=0.4, save_every=20)


def test_transport_coeffs():
    nu, eta = transport_coeffs(Re=1000, delta_u=2.0, length=10.0, Pm=1.0)
    assert nu == 2.0 * 10.0 / 1000
    assert eta == nu  # Pm = 1


def test_solver_is_stable_no_nan():
    out = _run(M_A=100.0)
    assert torch.isfinite(out["E_kin"]).all()
    assert torch.isfinite(out["omega_final"]).all()


def test_vorticity_stays_zero_mean():
    """The spectral scheme must not inject a spurious mean into a periodic field."""
    out = _run(M_A=100.0)
    assert out["omega_final"].mean().abs() < 1e-9


def test_weak_field_is_unstable():
    """Negligible magnetic field: the transverse kinetic energy E_y must grow."""
    out = _run(M_A=100.0)
    assert out["E_y"][-1] > 3.0 * out["E_y"][0]


def test_strong_field_is_stabilized():
    """Strong aligned field (M_A < 1): the instability is suppressed, E_y decays."""
    out = _run(M_A=0.7)
    assert out["E_y"][-1] < out["E_y"][0]
