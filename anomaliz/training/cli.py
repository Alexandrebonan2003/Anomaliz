from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config.settings import load_settings
from .pipeline import run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train an Anomaliz bundle.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML override.")
    parser.add_argument("--out", type=Path, required=True, help="Output bundle directory.", dest="out")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Also run capacity/window ablations and record them in metrics.json.",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    result = run_training(settings, args.out, sweep=args.sweep)
    print(f"Bundle: {result.bundle_dir}")
    print(f"Metrics: {json.dumps(result.metrics, indent=2)}")


if __name__ == "__main__":
    main()
