"""Run the solver over a sweep and store the trajectories on disk.

For each RunSpec we:
  1. build the grid and the Kelvin-Helmholtz initial state,
  2. integrate to t_end, recording `n_snapshots` uniform-in-time frames of (omega, a),
  3. save the trajectory + metadata to one HDF5 file in `out_dir`.

Design choices
--------------
- **One HDF5 file per run.** Simple, and the whole job is *restartable*: a run whose
  file already exists is skipped, so you can stop and resume.
- **float32 on disk.** Half the size of float64 and plenty for training; the physics
  was computed in float64.
- A JSON **manifest** lists every run and its parameters for the DataLoader to read.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

from ..solver.spectral import SpectralGrid
from ..solver.initial import kelvin_helmholtz_state
from ..solver.mhd2d import MHD2DSolver, transport_coeffs
from .sweep import RunSpec


def run_one(
    spec: RunSpec,
    out_dir: Path,
    t_end: float,
    n_snapshots: int,
    delta_u: float = 1.0,
    Pm: float = 1.0,
    shear_thickness: float | None = None,
    perturbation_amp: float = 1e-3,
    cfl: float = 0.25,
    overwrite: bool = False,
) -> dict:
    """Run a single simulation and write its HDF5 file. Returns the manifest entry."""
    out_path = out_dir / f"{spec.run_id}.h5"
    if out_path.exists() and not overwrite:
        return _manifest_entry(spec, out_path, skipped=True)

    grid = SpectralGrid(n=spec.resolution, dtype=torch.float64)
    nu, eta = transport_coeffs(Re=spec.Re, delta_u=delta_u, length=grid.length, Pm=Pm)
    state = kelvin_helmholtz_state(
        grid,
        M_A=spec.M_A,
        delta_u=delta_u,
        shear_thickness=shear_thickness,
        perturbation_amp=perturbation_amp,
        seed=spec.seed,
    )
    solver = MHD2DSolver(grid, nu=nu, eta=eta, B0=state.B0)

    t0 = time.time()
    out = solver.run(state, t_end=t_end, cfl=cfl, record_fields=True, n_snapshots=n_snapshots)
    wall = time.time() - t0

    with h5py.File(out_path, "w") as f:
        f.create_dataset("omega", data=out["omega_snaps"].to(torch.float32).numpy())
        f.create_dataset("a", data=out["a_snaps"].to(torch.float32).numpy())
        f.create_dataset("times", data=out["snap_times"].to(torch.float32).numpy())
        f.create_dataset("E_y", data=out["E_y"].to(torch.float32).numpy())
        # metadata as attributes
        for k, v in {
            "run_id": spec.run_id, "split": spec.split, "M_A": spec.M_A, "Re": spec.Re,
            "B0": state.B0, "nu": nu, "eta": eta, "resolution": spec.resolution,
            "seed": spec.seed, "delta_u": delta_u, "Pm": Pm, "t_end": t_end,
            "n_snapshots": n_snapshots, "wall_seconds": wall,
        }.items():
            f.attrs[k] = v

    return _manifest_entry(spec, out_path, skipped=False, wall=wall)


def _manifest_entry(spec: RunSpec, path: Path, skipped: bool, wall: float | None = None) -> dict:
    entry = spec.as_dict()
    entry.update({"file": str(path.name), "skipped": skipped})
    if wall is not None:
        entry["wall_seconds"] = round(wall, 2)
    return entry


def generate_dataset(
    specs: list[RunSpec],
    out_dir: str | Path,
    t_end: float = 20.0,
    n_snapshots: int = 80,
    overwrite: bool = False,
    progress: bool = True,
    **run_kwargs,
) -> list[dict]:
    """Generate all runs in `specs`, writing HDF5 files and a manifest.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    n_done = 0
    bar = tqdm(specs, disable=not progress, unit="run", desc="Generating")
    for spec in bar:
        entry = run_one(
            spec, out_dir, t_end=t_end, n_snapshots=n_snapshots, overwrite=overwrite, **run_kwargs
        )
        manifest.append(entry)
        if not entry["skipped"]:
            n_done += 1
        tag = "skip" if entry["skipped"] else f"{entry.get('wall_seconds', 0):.1f}s"
        bar.set_postfix_str(
            f"{spec.run_id} M_A={spec.M_A:.2f} Re={spec.Re:.0f} ({tag}) | new={n_done}"
        )

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
