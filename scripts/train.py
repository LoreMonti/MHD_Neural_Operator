"""Train the FNO on a generated dataset.

Usage:
    python scripts/train.py --config configs/kh_baseline.yaml

Not implemented yet — Phase 2.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    raise NotImplementedError(
        f"Training not implemented yet (config: {args.config}). See ROADMAP.md, Phase 2."
    )


if __name__ == "__main__":
    main()
