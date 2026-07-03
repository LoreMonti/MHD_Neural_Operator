"""Tests for rollout mechanics and spectra (no trained model required)."""

import torch

from mhd_fno.data.normalization import Normalizer
from mhd_fno.evaluation.rollout import rollout, rollout_rel_l2, transverse_energy_series
from mhd_fno.evaluation.spectra import kinetic_energy_spectrum
from mhd_fno.models.fno import FNO2d


def _identity_normalizer():
    return Normalizer(
        field_mean=torch.zeros(2), field_std=torch.ones(2),
        param_mean=torch.zeros(2), param_std=torch.ones(2),
    )


def test_rollout_shape():
    model = FNO2d(modes=6, width=8, n_layers=2)
    norm = _identity_normalizer()
    omega0 = torch.randn(32, 32); a0 = torch.randn(32, 32)
    traj = rollout(model, norm, omega0, a0, torch.tensor([3.0, 1000.0]), n_steps=5)
    assert traj.shape == (6, 2, 32, 32)


def test_rollout_can_change_resolution():
    model = FNO2d(modes=6, width=8, n_layers=2)
    norm = _identity_normalizer()
    traj = rollout(model, norm, torch.randn(32, 32), torch.randn(32, 32),
                   torch.tensor([3.0, 1000.0]), n_steps=3, resolution=48)
    assert traj.shape == (4, 2, 48, 48)


def test_rollout_rel_l2_zero_for_identical():
    t = torch.randn(4, 2, 16, 16)
    assert torch.allclose(rollout_rel_l2(t, t), torch.zeros(4), atol=1e-6)


def test_transverse_energy_nonnegative():
    traj = torch.randn(3, 2, 32, 32)
    e = transverse_energy_series(traj)
    assert (e >= 0).all() and e.shape == (3,)


def test_spectrum_shape_and_positive():
    k, E = kinetic_energy_spectrum(torch.randn(64, 64))
    assert len(k) == len(E) == 31    # shells 1..N/2-1
    assert (E >= 0).all()
