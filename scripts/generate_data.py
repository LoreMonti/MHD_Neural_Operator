"""Generate the ground-truth dataset by running the solver over the parameter sweep.

Usage:
    python scripts/generate_data.py --config configs/kh_baseline.yaml
    python scripts/generate_data.py --config configs/kh_baseline.yaml --n-train 8 --smoke

`--smoke` shrinks the job (few short low-res runs) to validate the pipeline quickly.
"""

from __future__ import annotations

import argparse

import yaml

from mhd_fno.data.generate import generate_dataset
from mhd_fno.data.sweep import generate_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-train", type=int, default=None, help="Override number of training runs.")
    parser.add_argument("--n-test", type=int, default=None, help="Override number of test runs.")
    parser.add_argument("--smoke", action="store_true", help="Tiny, fast run to test the pipeline.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    phys, ds, solver = cfg["physics"], cfg["dataset"], cfg["solver"]

    resolution = 32 if args.smoke else solver["resolution"]
    n_train = args.n_train if args.n_train is not None else (4 if args.smoke else ds["n_simulations"])
    n_test = args.n_test if args.n_test is not None else (2 if args.smoke else 30)
    t_end = 2.0 if args.smoke else ds["t_end"]
    n_snapshots = 6 if args.smoke else ds["snapshots_per_run"]

    specs = generate_sweep(
        n_train=n_train,
        n_test=n_test,
        M_A_range=tuple(phys["M_A_range"]),
        M_A_hole=tuple(phys["M_A_test_hole"]),
        Re_range=tuple(phys["Re_range"]),
        resolution=resolution,
    )
    print(f"Generating {len(specs)} runs at {resolution}^2, t_end={t_end}, "
          f"{n_snapshots} snapshots -> {ds['out_dir']}")
    generate_dataset(
        specs,
        out_dir=ds["out_dir"],
        t_end=t_end,
        n_snapshots=n_snapshots,
        Pm=phys["Pm"],
        shear_thickness=phys["shear_thickness"],
        perturbation_amp=phys["perturbation_amp"],
        cfl=solver["cfl"],
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
