"""Tests for the growth-rate fitter and theory helpers."""

import math

import torch

from mhd_fno.evaluation.growth import (
    growth_rate,
    michalke_max_growth,
    michalke_max_wavenumber,
)


def test_recovers_known_growth_rate():
    """energy = exp(2*gamma*t) must be fitted back to gamma."""
    gamma_true = 0.37
    t = torch.linspace(0, 10, 200)
    energy = torch.exp(2 * gamma_true * t)
    res = growth_rate(t, energy, window=(2.0, 8.0))
    assert math.isclose(res["gamma"], gamma_true, rel_tol=1e-6)
    assert res["r2"] > 0.999


def test_auto_window_on_transient_plus_saturation():
    """A curve that grows then saturates: the auto-window should still recover gamma."""
    gamma_true = 0.5
    t = torch.linspace(0, 20, 400)
    # logistic-like: exponential growth then saturation
    raw = torch.exp(2 * gamma_true * (t - 10))
    energy = raw / (1 + raw) + 1e-6
    res = growth_rate(t, energy)
    assert abs(res["gamma"] - gamma_true) < 0.1


def test_michalke_helpers():
    assert math.isclose(michalke_max_growth(1.0, 1.0), 0.1897)
    assert math.isclose(michalke_max_growth(2.0, 0.5), 0.1897 * 2.0 / 0.5)
    assert math.isclose(michalke_max_wavenumber(0.5), 0.4446 / 0.5)
