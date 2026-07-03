"""Loss functions for training the operator."""

from __future__ import annotations

import torch


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean over the batch of ||pred - target|| / ||target|| (per-sample relative error).

    Preferred over plain MSE for PDE fields: it is scale-invariant, so samples with
    large- and small-amplitude fields contribute comparably.
    """
    diff = (pred - target).flatten(1).norm(dim=1)
    denom = target.flatten(1).norm(dim=1).clamp_min(eps)
    return (diff / denom).mean()
