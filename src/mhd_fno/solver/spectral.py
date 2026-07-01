"""Spectral toolbox for the 2D pseudo-spectral MHD solver.

Everything here rests on one idea: on a periodic square box, any field can be written
as a sum of sine/cosine waves (a 2D Fourier series). In that representation spatial
derivatives become simple multiplications:

    d/dx  ->  multiply by (i * kx)
    d/dy  ->  multiply by (i * ky)
    laplacian (d2/dx2 + d2/dy2)  ->  multiply by -(kx^2 + ky^2) = -k2

So we move a field to Fourier space with an FFT, do calculus by multiplying, and come
back with an inverse FFT. This module provides the wavenumber grids and the derivative
/ inversion helpers; the solver in `mhd2d.py` uses them.

Conventions
-----------
- Fields are real tensors of shape (N, N) on a periodic box of side `length`.
- "hat" variables (e.g. `omega_hat`) live in Fourier space (complex, shape (N, N)).
- Velocity and magnetic field come from scalar potentials:
      v = curl(psi * z_hat)  =>  (v_x, v_y) = ( d psi/dy, -d psi/dx)
      B = curl(A   * z_hat)  =>  (B_x, B_y) = ( d A  /dy, -d A  /dx)
  which makes them divergence-free by construction.
- Vorticity: omega = -laplacian(psi)  =>  psi_hat = omega_hat / k2  (k=0 mode set to 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import math
import torch


@dataclass
class SpectralGrid:
    """Wavenumber grids and spectral operators for an N x N periodic box.

    Parameters
    ----------
    n : int
        Number of grid points per dimension.
    length : float
        Physical side length of the (square) periodic box. Defaults to 2*pi.
    device, dtype : torch options for the real-space fields.
    """

    n: int
    length: float = 2.0 * math.pi
    device: torch.device | str = "cpu"
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        n, length = self.n, self.length
        # Angular wavenumbers: fftfreq gives cycles per unit length; *2pi -> radians.
        dx = length / n
        k = 2.0 * math.pi * torch.fft.fftfreq(n, d=dx, device=self.device, dtype=self.dtype)
        # Broadcast to 2D: kx varies along axis 0, ky along axis 1.
        self.kx = k.view(n, 1).expand(n, n).contiguous()
        self.ky = k.view(1, n).expand(n, n).contiguous()
        self.k2 = self.kx**2 + self.ky**2
        # Inverse Laplacian multiplier 1/k2, with the k=0 (mean) mode set to 0 to avoid
        # dividing by zero. The mean of a periodic vorticity field carries no velocity.
        self.inv_k2 = torch.zeros_like(self.k2)
        nonzero = self.k2 > 0
        self.inv_k2[nonzero] = 1.0 / self.k2[nonzero]
        # 2/3 dealiasing mask: zero the top third of modes in each direction so that
        # quadratic nonlinear products don't alias energy back onto resolved modes.
        k_max = k.abs().max()
        cutoff = (2.0 / 3.0) * k_max
        self.dealias_mask = ((self.kx.abs() <= cutoff) & (self.ky.abs() <= cutoff)).to(self.dtype)

    # --- transforms -----------------------------------------------------------------
    def fft(self, field: torch.Tensor) -> torch.Tensor:
        """Real field -> Fourier coefficients."""
        return torch.fft.fft2(field)

    def ifft(self, field_hat: torch.Tensor) -> torch.Tensor:
        """Fourier coefficients -> real field (imaginary part discarded)."""
        return torch.fft.ifft2(field_hat).real

    # --- calculus in Fourier space --------------------------------------------------
    def grad(self, field_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Spectral gradient: returns (d/dx, d/dy) as Fourier coefficients."""
        return 1j * self.kx * field_hat, 1j * self.ky * field_hat

    def laplacian(self, field_hat: torch.Tensor) -> torch.Tensor:
        """Spectral Laplacian: multiply by -k2."""
        return -self.k2 * field_hat

    def inverse_laplacian(self, field_hat: torch.Tensor) -> torch.Tensor:
        """Solve laplacian(u) = field for u in Fourier space (zero-mean solution)."""
        return -self.inv_k2 * field_hat

    def dealias(self, field_hat: torch.Tensor) -> torch.Tensor:
        """Apply the 2/3-rule mask."""
        return field_hat * self.dealias_mask

    # --- potentials -> physical fields ----------------------------------------------
    def psi_hat_from_omega_hat(self, omega_hat: torch.Tensor) -> torch.Tensor:
        """Streamfunction from vorticity: omega = -lap(psi) => psi_hat = omega_hat / k2."""
        return self.inv_k2 * omega_hat

    def velocity_from_psi_hat(self, psi_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(v_x, v_y) = (d psi/dy, -d psi/dx), returned as real fields."""
        dpsi_dx_hat, dpsi_dy_hat = self.grad(psi_hat)
        v_x = self.ifft(dpsi_dy_hat)
        v_y = self.ifft(-dpsi_dx_hat)
        return v_x, v_y

    def field_from_potential_hat(self, pot_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(F_x, F_y) = (d pot/dy, -d pot/dx). Used for both v (from psi) and B (from A)."""
        dpot_dx_hat, dpot_dy_hat = self.grad(pot_hat)
        f_x = self.ifft(dpot_dy_hat)
        f_y = self.ifft(-dpot_dx_hat)
        return f_x, f_y
