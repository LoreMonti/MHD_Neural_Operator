"""Correctness tests for the spectral toolbox.

We test the operators against functions whose derivatives we know analytically.
On a 2*pi periodic box, take f(x, y) = sin(x) * cos(y). Then:
    df/dx = cos(x) cos(y),  df/dy = -sin(x) sin(y),  lap(f) = -2 f.
"""

import math

import torch

from mhd_fno.solver.spectral import SpectralGrid


def _sample_field(n: int):
    grid = SpectralGrid(n=n, length=2 * math.pi, dtype=torch.float64)
    # Build coordinates directly in float64: linspace/arange default to float32,
    # and a later .to(float64) would only preserve the float32-level error, leaving
    # the sample points ~1e-8 off the true grid and spoiling the comparison.
    x = torch.arange(n, dtype=torch.float64) * (2 * math.pi / n)
    xx = x.view(n, 1).expand(n, n)
    yy = x.view(1, n).expand(n, n)
    f = torch.sin(xx) * torch.cos(yy)
    return grid, xx, yy, f


def test_gradient_matches_analytic():
    grid, xx, yy, f = _sample_field(64)
    f_hat = grid.fft(f)
    dfdx_hat, dfdy_hat = grid.grad(f_hat)
    dfdx = grid.ifft(dfdx_hat)
    dfdy = grid.ifft(dfdy_hat)
    assert torch.allclose(dfdx, torch.cos(xx) * torch.cos(yy), atol=1e-10)
    assert torch.allclose(dfdy, -torch.sin(xx) * torch.sin(yy), atol=1e-10)


def test_laplacian_matches_analytic():
    grid, _, _, f = _sample_field(64)
    lap = grid.ifft(grid.laplacian(grid.fft(f)))
    assert torch.allclose(lap, -2.0 * f, atol=1e-10)


def test_inverse_laplacian_is_inverse():
    """inverse_laplacian(laplacian(f)) recovers f up to its (removed) mean."""
    grid, _, _, f = _sample_field(64)
    f = f - f.mean()  # zero-mean so the k=0 mode carries no information
    recovered = grid.ifft(grid.inverse_laplacian(grid.laplacian(grid.fft(f))))
    assert torch.allclose(recovered, f, atol=1e-10)


def test_velocity_from_psi_is_divergence_free():
    """v = curl(psi z) must satisfy div(v) = dv_x/dx + dv_y/dy = 0."""
    grid, _, _, psi = _sample_field(64)
    psi_hat = grid.fft(psi)
    v_x, v_y = grid.velocity_from_psi_hat(psi_hat)
    dvx_dx = grid.ifft(grid.grad(grid.fft(v_x))[0])
    dvy_dy = grid.ifft(grid.grad(grid.fft(v_y))[1])
    div = dvx_dx + dvy_dy
    assert div.abs().max() < 1e-10


def test_dealias_mask_zeros_high_modes():
    grid = SpectralGrid(n=48, dtype=torch.float64)
    # Every retained mode has |kx|,|ky| within 2/3 of k_max; masked ones are zeroed.
    assert grid.dealias_mask.max() == 1.0
    assert grid.dealias_mask.min() == 0.0
    ones = torch.ones(48, 48, dtype=torch.complex128)
    masked = grid.dealias(ones)
    assert torch.equal(masked.real, grid.dealias_mask)
