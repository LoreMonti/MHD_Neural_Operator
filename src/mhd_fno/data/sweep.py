"""Design the parameter sweep: which simulations to run.

We want hundreds of Kelvin-Helmholtz runs spread over the physical parameters
(M_A, Re). Two ideas drive the design:

1. Latin-Hypercube sampling (LHS) instead of a regular grid. A grid wastes runs (many
   share the same M_A or Re); LHS spreads samples so every 1D projection is evenly
   covered with far fewer points.

2. A held-out "threshold hole": training runs avoid M_A in [1.5, 2.5], and a separate
   batch of test runs lives *inside* that hole. If the FNO later reproduces the
   stabilization threshold there, it learned physics rather than interpolation.

Each run is described by a RunSpec (parameters + a split label + a seed).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from scipy.stats import qmc


@dataclass
class RunSpec:
    """One simulation to run."""

    run_id: str
    M_A: float
    Re: float
    resolution: int
    seed: int
    split: str          # "train" or "test"

    def as_dict(self) -> dict:
        return asdict(self)


def _lhs_unit(n: int, d: int, seed: int):
    """n points in the d-dim unit cube via Latin-Hypercube sampling."""
    sampler = qmc.LatinHypercube(d=d, seed=seed)
    return sampler.random(n)


def _map_M_A_with_hole(u, lo, hi, hole):
    """Map u in [0,1] onto [lo, hi] minus the open interval `hole`, preserving spread.

    The allowed set is two segments [lo, hole0] and [hole1, hi]; we place u by arc
    length along their union so samples never fall in the hole.
    """
    hole0, hole1 = hole
    left_len = hole0 - lo
    right_len = hi - hole1
    total = left_len + right_len
    pos = u * total
    return lo + pos if pos <= left_len else hole1 + (pos - left_len)


def generate_sweep(
    n_train: int = 250,
    n_test: int = 30,
    M_A_range: tuple[float, float] = (0.5, 6.0),
    M_A_hole: tuple[float, float] = (1.5, 2.5),
    Re_range: tuple[float, float] = (500.0, 5000.0),
    resolution: int = 128,
    seed: int = 0,
) -> list[RunSpec]:
    """Build the list of runs.

    Training runs: LHS over (M_A, Re) with M_A avoiding the hole.
    Test runs:     LHS over (M_A in the hole, Re) — the threshold-recovery set.
    """
    specs: list[RunSpec] = []

    # --- training runs (M_A avoids the hole) ---
    u_train = _lhs_unit(n_train, 2, seed=seed)
    for i, (u_ma, u_re) in enumerate(u_train):
        M_A = _map_M_A_with_hole(float(u_ma), M_A_range[0], M_A_range[1], M_A_hole)
        Re = Re_range[0] + float(u_re) * (Re_range[1] - Re_range[0])
        specs.append(RunSpec(f"train_{i:04d}", M_A, Re, resolution, seed=1000 + i, split="train"))

    # --- test runs (M_A inside the hole) ---
    u_test = _lhs_unit(n_test, 2, seed=seed + 1)
    for i, (u_ma, u_re) in enumerate(u_test):
        M_A = M_A_hole[0] + float(u_ma) * (M_A_hole[1] - M_A_hole[0])
        Re = Re_range[0] + float(u_re) * (Re_range[1] - Re_range[0])
        specs.append(RunSpec(f"test_{i:04d}", M_A, Re, resolution, seed=9000 + i, split="test"))

    return specs
