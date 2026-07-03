"""Autoregressive rollout of the FNO and comparison to the ground-truth trajectory.

One-step accuracy is easy; the real test is feeding the model its own output over and
over. Small per-step errors can compound, so a good operator must stay faithful over a
long rollout. Because the FNO is resolution-independent, a model trained at 64^2 can be
rolled out at the native 128^2 (the same normalization stats apply per pixel).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import torch
import torch.nn.functional as F

from ..data.normalization import Normalizer
from ..solver.spectral import SpectralGrid


def load_run(root: str | Path, run_id: str) -> dict:
    """Load a full ground-truth trajectory and its metadata from an HDF5 file."""
    root = Path(root)
    with h5py.File(root / f"{run_id}.h5", "r") as f:
        return {
            "omega": torch.from_numpy(f["omega"][:]),   # (T, H, W)
            "a": torch.from_numpy(f["a"][:]),
            "times": torch.from_numpy(f["times"][:]),
            "M_A": float(f.attrs["M_A"]),
            "Re": float(f.attrs["Re"]),
        }


@torch.no_grad()
def rollout(
    model: torch.nn.Module,
    normalizer: Normalizer,
    omega0: torch.Tensor,
    a0: torch.Tensor,
    params: torch.Tensor,
    n_steps: int,
    resolution: int | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Roll the model forward n_steps from the initial fields.

    Returns a trajectory tensor of shape (n_steps + 1, 2, H, W) in physical units.
    """
    model.eval()
    fields = torch.stack([omega0, a0], dim=0)[None].to(device).float()   # (1, 2, H, W)
    if resolution is not None and fields.shape[-1] != resolution:
        fields = F.interpolate(fields, size=(resolution, resolution), mode="bilinear",
                               align_corners=False)
    p = params[None].to(device).float()

    traj = [fields.clone()]
    for _ in range(n_steps):
        xn = normalizer.normalize_fields(fields)
        pred = normalizer.denormalize_fields(model(xn, normalizer.normalize_params(p)))
        fields = pred
        traj.append(fields.clone())
    return torch.cat(traj, dim=0).cpu()   # (n_steps + 1, 2, H, W)


def rollout_rel_l2(pred_traj: torch.Tensor, true_traj: torch.Tensor) -> torch.Tensor:
    """Per-step relative-L2 error between predicted and true trajectories."""
    diff = (pred_traj - true_traj).flatten(1).norm(dim=1)
    denom = true_traj.flatten(1).norm(dim=1).clamp_min(1e-12)
    return diff / denom


def transverse_energy_series(traj: torch.Tensor) -> torch.Tensor:
    """E_y(t) = 0.5 * mean(v_y^2) along a trajectory of (omega, a) fields.

    v_y = -d psi/dx with psi = inverse-Laplacian of omega; computed spectrally.
    """
    n = traj.shape[-1]
    grid = SpectralGrid(n=n, dtype=torch.float64)
    e_y = []
    for frame in traj:
        omega = frame[0].double()
        psi_hat = grid.psi_hat_from_omega_hat(grid.fft(omega))
        _, v_y = grid.velocity_from_psi_hat(psi_hat)
        e_y.append(0.5 * float((v_y**2).mean()))
    return torch.tensor(e_y)
