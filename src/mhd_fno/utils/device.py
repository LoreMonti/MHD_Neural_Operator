"""Pick the best available compute device: CUDA GPU, then Apple MPS, then CPU."""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> str:
    """Return the best device string. Pass `prefer` to force one (e.g. "cpu")."""
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
