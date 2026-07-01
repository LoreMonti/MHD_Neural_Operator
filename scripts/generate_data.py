"""Generate the ground-truth dataset by running the pseudo-spectral solver over the
parameter sweep defined in a config file.

Usage:
    python scripts/generate_data.py --config configs/kh_baseline.yaml

Not implemented yet — Phase 1.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    raise NotImplementedError(
        f"Data generation not implemented yet (config: {args.config}). See ROADMAP.md, Phase 1."
    )


if __name__ == "__main__":
    main()
