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


def fluctuation(field: torch.Tensor) -> torch.Tensor:
    """Remove the base flow: subtract the mean along the flow direction (x = axis -2).

    field - <field>_x isolates the x-dependent perturbation (the KH mode) from the
    x-independent base shear layer. This is a Reynolds decomposition.
    """
    return field - field.mean(dim=-2, keepdim=True)


def fluctuation_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative-L2 error measured on the perturbation only (base flow removed).

    Because it is scale-invariant per sample, a tiny early-time perturbation and a
    large saturated one contribute equally — forcing the model to learn the growing
    mode that the full-field loss ignores.
    """
    return relative_l2(fluctuation(pred), fluctuation(target))
