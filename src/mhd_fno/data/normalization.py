"""Field and parameter normalization.

Neural nets train best when their inputs are ~zero-mean, unit-variance. Our two field
channels (omega, a) and the two parameters (M_A, Re) live on very different scales, so
we standardize each one: subtract its mean, divide by its standard deviation. The same
statistics (computed once on the training set) are used to normalize inputs and targets
and to de-normalize the model's predictions back to physical units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch


@dataclass
class Normalizer:
    field_mean: torch.Tensor   # (C,)
    field_std: torch.Tensor    # (C,)
    param_mean: torch.Tensor   # (P,)
    param_std: torch.Tensor    # (P,)

    def normalize_fields(self, x: torch.Tensor) -> torch.Tensor:
        m = self.field_mean.view(1, -1, 1, 1).to(x.device)
        s = self.field_std.view(1, -1, 1, 1).to(x.device)
        return (x - m) / s

    def denormalize_fields(self, x: torch.Tensor) -> torch.Tensor:
        m = self.field_mean.view(1, -1, 1, 1).to(x.device)
        s = self.field_std.view(1, -1, 1, 1).to(x.device)
        return x * s + m

    def normalize_params(self, p: torch.Tensor) -> torch.Tensor:
        m = self.param_mean.view(1, -1).to(p.device)
        s = self.param_std.view(1, -1).to(p.device)
        return (p - m) / s

    # --- persistence ---
    def save(self, path: str | Path) -> None:
        obj = {k: getattr(self, k).tolist() for k in
               ("field_mean", "field_std", "param_mean", "param_std")}
        Path(path).write_text(json.dumps(obj))

    @classmethod
    def load(cls, path: str | Path) -> Normalizer:
        obj = json.loads(Path(path).read_text())
        return cls(**{k: torch.tensor(v) for k, v in obj.items()})


def compute_stats(root: str | Path, split: str = "train", max_files: int = 60,
                  seed: int = 0) -> Normalizer:
    """Estimate per-channel field stats and per-parameter stats from a file sample."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    files = [e for e in manifest if e["split"] == split]
    rng = np.random.default_rng(seed)
    if len(files) > max_files:
        files = [files[i] for i in rng.choice(len(files), max_files, replace=False)]

    # Welford-free: accumulate sums over sampled frames (cheap and sufficient).
    n = 0
    s1 = np.zeros(2)
    s2 = np.zeros(2)
    params = []
    for e in files:
        with h5py.File(root / e["file"], "r") as f:
            omega = f["omega"][:]
            a = f["a"][:]         # (T, H, W)
            for arr, ch in ((omega, 0), (a, 1)):
                s1[ch] += arr.sum()
                s2[ch] += (arr.astype(np.float64) ** 2).sum()
            n += omega.size
            params.append([float(f.attrs["M_A"]), float(f.attrs["Re"])])
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 1e-12)
    params = np.array(params)
    return Normalizer(
        field_mean=torch.tensor(mean, dtype=torch.float32),
        field_std=torch.tensor(np.sqrt(var), dtype=torch.float32),
        param_mean=torch.tensor(params.mean(0), dtype=torch.float32),
        param_std=torch.tensor(params.std(0) + 1e-8, dtype=torch.float32),
    )
