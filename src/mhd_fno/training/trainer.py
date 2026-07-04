"""Training loop with normalization, validation, and checkpointing.

The FNO predicts the *normalized* next state; we compare it to the normalized target
with a relative-L2 loss. Fields can optionally be downsampled (bilinear) to a smaller
training resolution, exploiting the FNO's resolution independence to cut CPU cost.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.normalization import Normalizer
from .losses import relative_l2, fluctuation_relative_l2


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        normalizer: Normalizer,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        train_resolution: int | None = None,
        fluct_weight: float = 0.0,
    ):
        self.model = model.to(device)
        self.norm = normalizer
        self.device = device
        self.train_resolution = train_resolution
        self.fluct_weight = fluct_weight   # weight of the perturbation-only loss term
        self.opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history: dict[str, list] = {"train": [], "val": []}
        self.best_val = float("inf")

    def _prep(self, batch):
        x = batch["input"].to(self.device).float()
        y = batch["target"].to(self.device).float()
        p = batch["params"].to(self.device).float()
        if self.train_resolution is not None and x.shape[-1] != self.train_resolution:
            n = self.train_resolution
            x = F.interpolate(x, size=(n, n), mode="bilinear", align_corners=False)
            y = F.interpolate(y, size=(n, n), mode="bilinear", align_corners=False)
        return self.norm.normalize_fields(x), self.norm.normalize_fields(y), self.norm.normalize_params(p)

    def _run_epoch(self, loader: DataLoader, train: bool) -> float:
        self.model.train(train)
        total, count = 0.0, 0
        for batch in loader:
            x, y, p = self._prep(batch)
            with torch.set_grad_enabled(train):
                pred = self.model(x, p)
                loss = relative_l2(pred, y)
                if self.fluct_weight > 0:
                    loss = loss + self.fluct_weight * fluctuation_relative_l2(pred, y)
                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
            total += loss.item() * x.size(0)
            count += x.size(0)
        return total / max(count, 1)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        epochs: int,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
    ) -> dict:
        for ep in range(epochs):
            t0 = time.time()
            tr = self._run_epoch(train_loader, train=True)
            va = self._run_epoch(val_loader, train=False) if val_loader is not None else float("nan")
            self.history["train"].append(tr)
            self.history["val"].append(va)
            if ckpt_path is not None and va < self.best_val:
                self.best_val = va
                self.save(ckpt_path, epoch=ep)
            if verbose:
                print(f"epoch {ep + 1}/{epochs}  train {tr:.4f}  val {va:.4f}  ({time.time() - t0:.1f}s)")
        return self.history

    # --- checkpointing ---
    def save(self, path: str | Path, epoch: int = -1) -> None:
        torch.save(
            {"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
             "history": self.history, "best_val": self.best_val, "epoch": epoch},
            path,
        )

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        self.history = ckpt["history"]
        self.best_val = ckpt["best_val"]
