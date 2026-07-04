"""Tests for the loss functions and the Reynolds decomposition."""

import torch

from mhd_fno.training.losses import relative_l2, fluctuation, fluctuation_relative_l2


def test_relative_l2_zero_and_scale_invariant():
    x = torch.randn(4, 2, 16, 16)
    assert relative_l2(x, x) < 1e-6
    # scaling both pred error and target by the same factor leaves the ratio unchanged
    a = relative_l2(x + 0.1, x)
    b = relative_l2(10 * (x + 0.1), 10 * x)
    assert torch.isclose(a, b, atol=1e-6)


def test_fluctuation_removes_x_mean():
    field = torch.randn(2, 2, 8, 8)
    f = fluctuation(field)
    # mean along x (axis -2) must be ~0 everywhere
    assert f.mean(dim=-2).abs().max() < 1e-6


def test_fluctuation_ignores_pure_base_flow():
    """A field that is purely x-independent (base flow) has zero fluctuation."""
    base = torch.randn(1, 1, 1, 8).expand(1, 1, 8, 8).contiguous()  # depends only on y
    assert fluctuation(base).abs().max() < 1e-6


def test_fluctuation_loss_focuses_on_perturbation():
    """If pred matches the base but not the tiny perturbation, full-field loss is small
    but fluctuation loss is large."""
    base = torch.zeros(1, 1, 16, 16)
    base[..., :, :8] = 1.0  # a big step along y only -> x-independent (the 'river')
    pert = 1e-3 * torch.randn(1, 1, 16, 16)  # tiny x-dependent 'apple'
    target = base + pert
    pred = base.clone()  # gets the river, misses the apple
    assert relative_l2(pred, target) < 1e-2                 # full-field: looks fine
    assert fluctuation_relative_l2(pred, target) > 0.9      # perturbation: clearly wrong
