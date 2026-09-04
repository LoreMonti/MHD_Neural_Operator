"""Tests for the Kelvin-Helmholtz initial condition."""

import math

import torch

from mhd_fno.solver.initial import B0_from_M_A, kelvin_helmholtz_state
from mhd_fno.solver.spectral import SpectralGrid


def test_B0_from_M_A():
    # Normalized units: v_A = B0, so B0 = delta_u / M_A.
    assert math.isclose(B0_from_M_A(2.0, delta_u=1.0), 0.5)
    assert math.isclose(B0_from_M_A(0.5, delta_u=1.0), 2.0)


def test_vorticity_is_zero_mean_and_periodic():
    grid = SpectralGrid(n=128, dtype=torch.float64)
    st = kelvin_helmholtz_state(grid, M_A=2.0, seed=0)
    # A periodic vorticity field must have (near) zero spatial mean.
    assert st.omega.mean().abs() < 1e-10
    assert st.omega.shape == (128, 128)


def test_velocity_reconstructs_shear_profile():
    """Rebuild v from omega and check it matches the intended double shear layer."""
    grid = SpectralGrid(n=128, dtype=torch.float64)
    delta_u = 1.0
    st = kelvin_helmholtz_state(grid, M_A=2.0, delta_u=delta_u, perturbation_amp=0.0, seed=0)
    psi_hat = grid.psi_hat_from_omega_hat(grid.fft(st.omega))
    v_x, v_y = grid.velocity_from_psi_hat(psi_hat)
    # With no perturbation, v_x should span roughly [-delta_u/2, +delta_u/2].
    assert math.isclose(v_x.max().item(), 0.5 * delta_u, abs_tol=2e-2)
    assert math.isclose(v_x.min().item(), -0.5 * delta_u, abs_tol=2e-2)
    # v_y (from the reconstructed streamfunction) is essentially zero without a seed.
    assert v_y.abs().max() < 1e-3


def test_perturbation_amplitude_scales():
    grid = SpectralGrid(n=128, dtype=torch.float64)
    delta_u, amp = 1.0, 1e-3
    st = kelvin_helmholtz_state(grid, M_A=2.0, delta_u=delta_u, perturbation_amp=amp, seed=1)
    # Reconstruct v_y and check its peak is ~ amp * delta_u (the requested seed size).
    psi_hat = grid.psi_hat_from_omega_hat(grid.fft(st.omega))
    _, v_y = grid.velocity_from_psi_hat(psi_hat)
    assert 0.2 * amp * delta_u < v_y.abs().max().item() < 5 * amp * delta_u


def test_seed_is_reproducible_and_varies():
    grid = SpectralGrid(n=64, dtype=torch.float64)
    a0 = kelvin_helmholtz_state(grid, M_A=2.0, seed=0).omega
    a0_again = kelvin_helmholtz_state(grid, M_A=2.0, seed=0).omega
    a1 = kelvin_helmholtz_state(grid, M_A=2.0, seed=1).omega
    assert torch.equal(a0, a0_again)          # same seed -> identical
    assert not torch.equal(a0, a1)            # different seed -> different perturbation
