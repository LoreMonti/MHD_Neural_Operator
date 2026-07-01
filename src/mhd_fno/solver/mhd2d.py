"""Pseudo-spectral 2D incompressible MHD solver.

This is the engine: it takes the starting state and advances it in time according to
the MHD equations. We work with two scalar fields on a periodic box,

    omega   vorticity of the flow          (v = curl(psi z),  omega = -lap psi)
    a       fluctuating magnetic potential  (A = B0*y + a,  B = curl(A z))

and the aligned mean field B0 as a parameter. In normalized units (rho = 1, mu0 = 1)
the equations are

    d(omega)/dt + (v . grad) omega = (B . grad) j + nu  * lap(omega)     (vorticity)
    d(a)/dt     + (v . grad) a      = -B0 * v_y          + eta * lap(a)   (induction)

with the current density j = -lap(A) = -lap(a) and B = (B0 + da/dy, -da/dx).

"Pseudo-spectral" = derivatives are computed in Fourier space (exact), but the
nonlinear products (v . grad) are formed in real space (cheap), then transformed back
and de-aliased with the 2/3 rule to remove the spurious high modes those products
create.

Time stepping is classic explicit RK4 with a CFL-limited step. Viscosity nu and
resistivity eta are the linear diffusion terms; they are included in the RHS directly
(explicit), which is stable here because the advective CFL is the tighter limit.
"""

from __future__ import annotations

import math
import torch

from .spectral import SpectralGrid
from .initial import KHState


def transport_coeffs(Re: float, delta_u: float, length: float, Pm: float = 1.0) -> tuple[float, float]:
    """Viscosity and resistivity from Reynolds number and magnetic Prandtl number.

    Re = delta_u * L / nu  =>  nu = delta_u * L / Re;   Pm = nu / eta  =>  eta = nu / Pm.
    """
    nu = delta_u * length / Re
    eta = nu / Pm
    return nu, eta


class MHD2DSolver:
    """Advance (omega, a) in time for 2D incompressible MHD."""

    def __init__(self, grid: SpectralGrid, nu: float, eta: float, B0: float) -> None:
        self.grid = grid
        self.nu = nu
        self.eta = eta
        self.B0 = B0

    # --- physical fields from (omega, a) --------------------------------------------
    def fields(self, omega: torch.Tensor, a: torch.Tensor):
        """Return (v_x, v_y, b_x, b_y, j) as real-space fields."""
        g = self.grid
        omega_hat, a_hat = g.fft(omega), g.fft(a)
        psi_hat = g.psi_hat_from_omega_hat(omega_hat)
        v_x, v_y = g.velocity_from_psi_hat(psi_hat)
        dadx_hat, dady_hat = g.grad(a_hat)
        b_x = self.B0 + g.ifft(dady_hat)     # B_x = B0 + da/dy
        b_y = g.ifft(-dadx_hat)              # B_y = -da/dx
        j = g.ifft(g.k2 * a_hat)             # j = -lap(a)  ->  j_hat = k2 * a_hat
        return v_x, v_y, b_x, b_y, j

    # --- right-hand side of the PDE -------------------------------------------------
    def rhs(self, omega: torch.Tensor, a: torch.Tensor):
        """Time derivatives (d omega/dt, d a/dt) as real-space fields."""
        g = self.grid
        omega_hat, a_hat = g.fft(omega), g.fft(a)
        psi_hat = g.psi_hat_from_omega_hat(omega_hat)

        # velocity and magnetic field
        v_x, v_y = g.velocity_from_psi_hat(psi_hat)
        dadx_hat, dady_hat = g.grad(a_hat)
        b_x = self.B0 + g.ifft(dady_hat)
        b_y = g.ifft(-dadx_hat)

        # current and the spatial gradients we need (all spectral -> real)
        j_hat = g.k2 * a_hat
        djdx_hat, djdy_hat = g.grad(j_hat)
        domega_dx_hat, domega_dy_hat = g.grad(omega_hat)
        j = g.ifft(j_hat)  # noqa: F841  (kept for clarity / possible diagnostics)
        dj_dx, dj_dy = g.ifft(djdx_hat), g.ifft(djdy_hat)
        domega_dx, domega_dy = g.ifft(domega_dx_hat), g.ifft(domega_dy_hat)
        da_dx, da_dy = g.ifft(dadx_hat), g.ifft(dady_hat)

        # --- vorticity equation ---
        advect_omega = v_x * domega_dx + v_y * domega_dy      # (v . grad) omega
        lorentz = b_x * dj_dx + b_y * dj_dy                   # (B . grad) j
        nonlinear_omega_hat = g.dealias(g.fft(lorentz - advect_omega))
        domega_hat = nonlinear_omega_hat + self.nu * g.laplacian(omega_hat)

        # --- induction equation ---
        advect_a = v_x * da_dx + v_y * da_dy                  # (v . grad) a
        # -B0*v_y comes from advecting the mean-field part B0*y; it is linear in v_y.
        nonlinear_a_hat = g.dealias(g.fft(-advect_a - self.B0 * v_y))
        da_hat = nonlinear_a_hat + self.eta * g.laplacian(a_hat)

        return g.ifft(domega_hat), g.ifft(da_hat)

    # --- time stepping --------------------------------------------------------------
    def step(self, omega: torch.Tensor, a: torch.Tensor, dt: float):
        """One classic RK4 step."""
        k1o, k1a = self.rhs(omega, a)
        k2o, k2a = self.rhs(omega + 0.5 * dt * k1o, a + 0.5 * dt * k1a)
        k3o, k3a = self.rhs(omega + 0.5 * dt * k2o, a + 0.5 * dt * k2a)
        k4o, k4a = self.rhs(omega + dt * k3o, a + dt * k3a)
        omega_new = omega + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
        a_new = a + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)
        return omega_new, a_new

    def compute_dt(self, omega: torch.Tensor, a: torch.Tensor, cfl: float = 0.4) -> float:
        """CFL-limited time step from the fastest signal speed (flow + Alfven)."""
        g = self.grid
        v_x, v_y, b_x, b_y, _ = self.fields(omega, a)
        dx = g.length / g.n
        max_speed = torch.sqrt(v_x**2 + v_y**2).max()
        max_alfven = torch.sqrt(b_x**2 + b_y**2).max()
        speed = float(max(max_speed, max_alfven)) + 1e-12
        dt_adv = cfl * dx / speed
        # diffusive limit (usually looser than advective here)
        diff = max(self.nu, self.eta) + 1e-12
        dt_diff = cfl * dx * dx / diff
        return min(dt_adv, dt_diff)

    # --- diagnostics ----------------------------------------------------------------
    def energies(self, omega: torch.Tensor, a: torch.Tensor) -> dict:
        """Box-averaged energies. E_y (transverse kinetic) is the clean growth signal."""
        v_x, v_y, b_x, b_y, _ = self.fields(omega, a)
        return {
            "E_kin": 0.5 * float((v_x**2 + v_y**2).mean()),
            "E_mag": 0.5 * float((b_x**2 + b_y**2).mean()),
            "E_y": 0.5 * float((v_y**2).mean()),   # ~0 initially, grows with the instability
        }

    # --- driver ---------------------------------------------------------------------
    def run(
        self,
        state: KHState,
        t_end: float,
        cfl: float = 0.4,
        save_every: int = 10,
        record_fields: bool = False,
    ) -> dict:
        """Integrate from `state` to time `t_end`.

        Returns a dict with time series of energies (always) and, if `record_fields`,
        snapshots of (omega, a). Snapshots/diagnostics are recorded every `save_every`
        steps.
        """
        omega, a = state.omega.clone(), state.a.clone()
        t = 0.0
        step_idx = 0
        times, e_kin, e_mag, e_y = [], [], [], []
        omega_snaps, a_snaps, snap_times = [], [], []

        def record():
            en = self.energies(omega, a)
            times.append(t)
            e_kin.append(en["E_kin"])
            e_mag.append(en["E_mag"])
            e_y.append(en["E_y"])
            if record_fields:
                omega_snaps.append(omega.clone())
                a_snaps.append(a.clone())
                snap_times.append(t)

        record()
        while t < t_end:
            dt = self.compute_dt(omega, a, cfl=cfl)
            dt = min(dt, t_end - t)
            omega, a = self.step(omega, a, dt)
            t += dt
            step_idx += 1
            if step_idx % save_every == 0:
                record()

        out = {
            "times": torch.tensor(times),
            "E_kin": torch.tensor(e_kin),
            "E_mag": torch.tensor(e_mag),
            "E_y": torch.tensor(e_y),
            "omega_final": omega,
            "a_final": a,
        }
        if record_fields:
            out["omega_snaps"] = torch.stack(omega_snaps)
            out["a_snaps"] = torch.stack(a_snaps)
            out["snap_times"] = torch.tensor(snap_times)
        return out
