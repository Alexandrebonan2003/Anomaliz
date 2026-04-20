from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config.settings import load_settings
from ..tracking.loggers import build_logger
from .pipeline import run_training


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train an Anomaliz bundle.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML override.")
    parser.add_argument("--out", type=Path, required=True, help="Output bundle directory.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Also run capacity/window ablations and record them in metrics.json.",
    )
    parser.add_argument(
        "--logger",
        choices=["noop", "mlflow"],
        default="noop",
        help="Experiment logger backend (default: noop).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="MLflow experiment name (default: from config).",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        dest="run_name",
        help="MLflow run name.",
    )
    parser.add_argument(
        "--tracking-uri",
        default=None,
        dest="tracking_uri",
        help="MLflow tracking URI (overrides config).",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)

    experiment_name = args.experiment or settings.tracking.experiment_name
    tracking_uri = args.tracking_uri or settings.tracking.tracking_uri

    logger = build_logger(
        args.logger,
        experiment_name=experiment_name,
        run_name=args.run_name,
        tracking_uri=tracking_uri,
    )

    result = run_training(settings, args.out, sweep=args.sweep, logger=logger)
    print(f"Bundle: {result.bundle_dir}")
    print(f"Metrics: {json.dumps(result.metrics, indent=2)}")


if __name__ == "__main__":
    main()
