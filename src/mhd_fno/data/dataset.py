"""PyTorch Dataset over the generated trajectories.

The FNO learns a one-step map field(t) -> field(t+dt). So from each trajectory of T
snapshots we make T-1 training pairs: (snapshot i) -> (snapshot i+1). A "field" here is
the 2-channel stack (omega, a).

Each item is:
    input  : (2, N, N)  = [omega(t),   a(t)]
    target : (2, N, N)  = [omega(t+dt), a(t+dt)]
    params : (2,)       = [M_A, Re]      (for conditioning the operator)

HDF5 files are opened lazily and cached per Dataset instance. For multi-worker loading
use a fresh Dataset per worker (or num_workers=0); handles are not shared across procs.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class MHDTrajectoryDataset(Dataset):
    def __init__(self, root: str | Path, split: str | None = None, pair_stride: int = 1,
                 include_runs: set[str] | None = None, rollout_steps: int = 1):
        """
        Parameters
        ----------
        root : directory containing the HDF5 files and manifest.json.
        split : "train", "test", or None (all).
        pair_stride : keep every k-th (run, time) pair. >1 shrinks the dataset for
            faster CPU training while still covering all runs and time ranges.
        include_runs : if given, keep only runs whose run_id is in this set (used to
            carve a validation set out of unseen training runs).
        rollout_steps : if >1, each item is a window of (rollout_steps + 1) consecutive
            frames for multi-step training; the item returns {"window", "params"}
            instead of {"input", "target", "params"}.
        """
        self.rollout_steps = rollout_steps
        self.root = Path(root)
        with open(self.root / "manifest.json") as f:
            manifest = json.load(f)
        self.entries = [
            e for e in manifest
            if (split is None or e["split"] == split)
            and (include_runs is None or e["run_id"] in include_runs)
        ]

        # Build a flat index of (entry_idx, t) pairs and read light metadata up front.
        self._handles: dict[str, h5py.File] = {}
        self.index: list[tuple[int, int]] = []
        self.params: list[tuple[float, float]] = []
        for ei, e in enumerate(self.entries):
            with h5py.File(self.root / e["file"], "r") as f:
                n_t = f["omega"].shape[0]
                m_a, re = float(f.attrs["M_A"]), float(f.attrs["Re"])
            for t in range(n_t - self.rollout_steps):
                self.index.append((ei, t))
                self.params.append((m_a, re))

        if pair_stride > 1:
            self.index = self.index[::pair_stride]
            self.params = self.params[::pair_stride]

    def _file(self, entry_idx: int) -> h5py.File:
        path = str(self.root / self.entries[entry_idx]["file"])
        h = self._handles.get(path)
        if h is None:
            h = h5py.File(path, "r")
            self._handles[path] = h
        return h

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        entry_idx, t = self.index[i]
        f = self._file(entry_idx)
        params = torch.tensor(self.params[i], dtype=torch.float32)
        if self.rollout_steps > 1:
            k = self.rollout_steps + 1
            omega = torch.from_numpy(f["omega"][t : t + k])      # (k, N, N)
            a = torch.from_numpy(f["a"][t : t + k])
            window = torch.stack([omega, a], dim=1)              # (k, 2, N, N)
            return {"window": window, "params": params}
        omega = torch.from_numpy(f["omega"][t : t + 2])   # (2, N, N): times t and t+1
        a = torch.from_numpy(f["a"][t : t + 2])
        x = torch.stack([omega[0], a[0]], dim=0)          # (2, N, N)
        y = torch.stack([omega[1], a[1]], dim=0)
        return {"input": x, "target": y, "params": params}

    def __del__(self):
        for h in getattr(self, "_handles", {}).values():
            try:
                h.close()
            except Exception:
                pass
