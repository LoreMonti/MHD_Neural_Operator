"""Tests for the FNO architecture."""

import torch

from mhd_fno.models.fno import FNO2d, SpectralConv2d


def test_output_shape():
    model = FNO2d(modes=8, width=16, n_layers=2)
    x = torch.randn(3, 2, 32, 32)
    p = torch.randn(3, 2)
    y = model(x, p)
    assert y.shape == (3, 2, 32, 32)


def test_runs_at_different_resolution():
    """Same weights must accept a different grid size (resolution independence)."""
    model = FNO2d(modes=8, width=16, n_layers=2)
    for n in (32, 48, 64):
        y = model(torch.randn(1, 2, n, n), torch.randn(1, 2))
        assert y.shape == (1, 2, n, n)


def test_spectral_conv_is_differentiable():
    conv = SpectralConv2d(4, 4, modes1=6, modes2=6)
    x = torch.randn(2, 4, 24, 24, requires_grad=True)
    conv(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_can_overfit_one_batch():
    """A working operator must be able to memorize a single input->target pair."""
    torch.manual_seed(0)
    model = FNO2d(modes=12, width=32, n_layers=3)
    x = torch.randn(2, 2, 32, 32)
    p = torch.randn(2, 2)
    target = torch.randn(2, 2, 32, 32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss0 = None
    for _ in range(200):
        opt.zero_grad()
        loss = ((model(x, p) - target) ** 2).mean()
        if loss0 is None:
            loss0 = loss.item()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * loss0   # loss dropped by >10x -> it learns
