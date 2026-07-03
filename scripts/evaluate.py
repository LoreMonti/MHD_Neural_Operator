"""Evaluate a trained FNO: rollout fidelity, energy spectra, and threshold recovery.

Produces (in --out):
  - rollout_error.png    : per-step relative-L2 error over a long rollout (a few runs)
  - rollout_snapshot.png : true vs FNO vorticity at a late time (one unstable run)
  - spectrum.png         : kinetic energy spectrum, true vs FNO
  - threshold.png        : growth rate gamma(M_A), true vs FNO, on the held-out test hole

Usage:
    python scripts/evaluate.py --config configs/kh_baseline.yaml --ckpt checkpoints/fno_best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from mhd_fno.models.fno import FNO2d
from mhd_fno.data.normalization import Normalizer
from mhd_fno.evaluation.rollout import load_run, rollout, rollout_rel_l2, transverse_energy_series
from mhd_fno.evaluation.spectra import kinetic_energy_spectrum
from mhd_fno.evaluation.growth import growth_rate, M_A_CRITICAL


def _resize_traj(traj, n):
    return F.interpolate(traj, size=(n, n), mode="bilinear", align_corners=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", default="checkpoints/fno_best.pt")
    p.add_argument("--normalizer", default="checkpoints/normalizer.json")
    p.add_argument("--data", default=None)
    p.add_argument("--out", default="notebooks")
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--eval-res", type=int, default=128, help="Rollout resolution (invariance test).")
    p.add_argument("--scan-res", type=int, default=64, help="Resolution for the threshold scan.")
    args = p.parse_args()

    cfg = __import__("yaml").safe_load(open(args.config))
    data_dir = Path(args.data or cfg["dataset"]["out_dir"])
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    norm = Normalizer.load(args.normalizer)
    model = FNO2d(modes=args.modes, width=args.width, n_layers=args.n_layers)
    model.load_state_dict(torch.load(args.ckpt)["model"])
    model.eval()

    manifest = json.loads((data_dir / "manifest.json").read_text())
    test = [e for e in manifest if e["split"] == "test"]

    # === Metric A: rollout error over time (resolution-invariance at eval_res) ===
    show = sorted(test, key=lambda e: e["M_A"])[:: max(1, len(test) // 3)][:3]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for e in show:
        run = load_run(data_dir, e["run_id"])
        n_steps = run["omega"].shape[0] - 1
        params = torch.tensor([run["M_A"], run["Re"]])
        pred = rollout(model, norm, run["omega"][0], run["a"][0], params, n_steps, resolution=args.eval_res)
        true = torch.stack([run["omega"], run["a"]], dim=1)          # (T, 2, H, W)
        true = _resize_traj(true, args.eval_res)
        err = rollout_rel_l2(pred, true)
        ax.plot(run["times"], err, label=f"M_A={run['M_A']:.2f}")
    ax.set(xlabel="t", ylabel="relative L2 error", title=f"Autoregressive rollout error ({args.eval_res}^2)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "rollout_error.png", dpi=110); plt.close(fig)

    # === Metric B & C: snapshot + spectrum for one unstable run ===
    e = max(test, key=lambda e: e["M_A"])
    run = load_run(data_dir, e["run_id"])
    n_steps = run["omega"].shape[0] - 1
    params = torch.tensor([run["M_A"], run["Re"]])
    pred = rollout(model, norm, run["omega"][0], run["a"][0], params, n_steps, resolution=args.eval_res)
    true = _resize_traj(torch.stack([run["omega"], run["a"]], dim=1), args.eval_res)
    ti = int(0.75 * n_steps)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for a_, (f_, t_) in zip(axs, [(true[ti, 0], "true omega"), (pred[ti, 0], "FNO omega"),
                                  (pred[ti, 0] - true[ti, 0], "error")]):
        im = a_.imshow(f_.numpy(), origin="lower", cmap="RdBu_r"); a_.set_title(t_); a_.axis("off")
        plt.colorbar(im, ax=a_, shrink=0.7)
    fig.suptitle(f"Vorticity at t={run['times'][ti]:.1f}  (M_A={run['M_A']:.2f})")
    fig.tight_layout(); fig.savefig(out / "rollout_snapshot.png", dpi=90); plt.close(fig)

    k_t, E_t = kinetic_energy_spectrum(true[ti, 0])
    k_p, E_p = kinetic_energy_spectrum(pred[ti, 0])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.loglog(k_t, E_t, label="true"); ax.loglog(k_p, E_p, "--", label="FNO")
    ax.set(xlabel="k", ylabel="E(k)", title=f"Kinetic energy spectrum at t={run['times'][ti]:.1f}")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "spectrum.png", dpi=110); plt.close(fig)

    # === Metric D: threshold recovery on the held-out hole ===
    print("threshold scan over test runs (this is the key test)...")
    rows = []
    for e in sorted(test, key=lambda e: e["M_A"]):
        run = load_run(data_dir, e["run_id"])
        n_steps = run["omega"].shape[0] - 1
        params = torch.tensor([run["M_A"], run["Re"]])
        pred = rollout(model, norm, run["omega"][0], run["a"][0], params, n_steps, resolution=args.scan_res)
        ey_pred = transverse_energy_series(pred)
        true = _resize_traj(torch.stack([run["omega"], run["a"]], dim=1), args.scan_res)
        ey_true = transverse_energy_series(true)
        g_pred = growth_rate(run["times"], ey_pred, window=(1.5, 8.0))["gamma"]
        g_true = growth_rate(run["times"], ey_true, window=(1.5, 8.0))["gamma"]
        rows.append((run["M_A"], g_true, g_pred))
        print(f"  M_A={run['M_A']:.2f}  gamma true={g_true:+.3f}  FNO={g_pred:+.3f}")

    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(M_A_CRITICAL, color="gray", ls="--", label="theory M_A=2")
    ax.plot(rows[:, 0], rows[:, 1], "o-", label="true (solver)")
    ax.plot(rows[:, 0], rows[:, 2], "s--", label="FNO")
    ax.set(xlabel="M_A", ylabel="growth rate gamma",
           title="Magnetic stabilization threshold — held-out test hole")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "threshold.png", dpi=110); plt.close(fig)
    print(f"saved figures to {out}/")


if __name__ == "__main__":
    main()
