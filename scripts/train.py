"""Train the FNO on the generated dataset.

Splits the training runs into train/validation (by run, so validation runs are unseen),
normalizes fields and parameters, and fits the operator with a relative-L2 loss. Saves
the best checkpoint, the normalizer, and a loss curve.

On a CPU it is much cheaper to train at a reduced resolution (the FNO is
resolution-independent) and to subsample time pairs. Defaults are tuned for that.

Usage:
    python scripts/train.py --config configs/kh_baseline.yaml
    python scripts/train.py --config configs/kh_baseline.yaml --epochs 60 --train-res 64
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from mhd_fno.data.dataset import MHDTrajectoryDataset
from mhd_fno.data.normalization import compute_stats
from mhd_fno.models.fno import FNO2d
from mhd_fno.training.trainer import Trainer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", default=None, help="Dataset dir (default: from config).")
    p.add_argument("--out", default="checkpoints", help="Output dir for checkpoint/curve.")
    p.add_argument("--train-res", type=int, default=64, help="Downsample training resolution.")
    p.add_argument("--modes", type=int, default=16)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--pair-stride", type=int, default=4, help="Keep every k-th time pair.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--fluct-weight", type=float, default=1.0,
                   help="Weight of the perturbation-only loss term (Phase 3). 0 = full-field only.")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = json.load(f) if args.config.endswith(".json") else __import__("yaml").safe_load(f)
    data_dir = args.data or cfg["dataset"]["out_dir"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # --- split training runs into train / val (by run) ---
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text())
    train_runs = [e["run_id"] for e in manifest if e["split"] == "train"]
    random.Random(args.seed).shuffle(train_runs)
    n_val = max(1, int(args.val_frac * len(train_runs)))
    val_ids, tr_ids = set(train_runs[:n_val]), set(train_runs[n_val:])
    print(f"train runs: {len(tr_ids)}  val runs: {len(val_ids)}")

    # --- data ---
    norm = compute_stats(data_dir, split="train")
    norm.save(out_dir / "normalizer.json")
    train_ds = MHDTrajectoryDataset(data_dir, split="train", pair_stride=args.pair_stride, include_runs=tr_ids)
    val_ds = MHDTrajectoryDataset(data_dir, split="train", pair_stride=args.pair_stride, include_runs=val_ids)
    print(f"train pairs: {len(train_ds)}  val pairs: {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # --- model & training ---
    model = FNO2d(modes=args.modes, width=args.width, n_layers=args.n_layers)
    n_par = sum(t.numel() for t in model.parameters())
    print(f"model: {n_par/1e6:.2f}M params, training at {args.train_res}^2")
    trainer = Trainer(model, norm, device=device, lr=args.lr, train_resolution=args.train_res,
                      fluct_weight=args.fluct_weight)
    print(f"fluctuation-loss weight: {args.fluct_weight}")
    trainer.fit(train_loader, val_loader, epochs=args.epochs, ckpt_path=out_dir / "fno_best.pt")

    # --- loss curve ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(trainer.history["train"], label="train")
    ax.plot(trainer.history["val"], label="val")
    ax.set(xlabel="epoch", ylabel="relative L2 loss", title="FNO training")
    ax.set_yscale("log"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=110)
    print(f"best val loss: {trainer.best_val:.4f}  ->  {out_dir/'fno_best.pt'}")


if __name__ == "__main__":
    main()
