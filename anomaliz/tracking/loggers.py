from __future__ import annotations

from pathlib import Path
from typing import Any


class NoOpLogger:
    """Silent logger — zero overhead, no external dependencies."""

    def log_params(self, params: dict[str, Any]) -> None:
        pass

    def log_metrics(self, metrics: dict[str, float]) -> None:
        pass

    def log_artifact(self, local_path: Path) -> None:
        pass

    def log_model(self, model: object, artifact_path: str) -> None:
        pass

    def __enter__(self) -> "NoOpLogger":
        return self

    def __exit__(self, *args: object) -> None:
        pass


class MLflowLogger:
    """MLflow-backed experiment logger.

    Wraps a single MLflow run as a context manager. The run is started on
    ``__enter__`` and ended on ``__exit__``, so the caller does::

        with MLflowLogger(tracking_uri="sqlite:///mlruns.db") as log:
            log.log_params(...)
            log.log_metrics(...)
    """

    def __init__(
        self,
        experiment_name: str = "anomaliz",
        run_name: str | None = None,
        tracking_uri: str | None = None,
    ) -> None:
        self._experiment_name = experiment_name
        self._run_name = run_name
        self._tracking_uri = tracking_uri

    def __enter__(self) -> "MLflowLogger":
        import mlflow

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        mlflow.set_experiment(self._experiment_name)
        mlflow.start_run(run_name=self._run_name)
        return self

    def __exit__(self, *args: object) -> None:
        import mlflow

        mlflow.end_run()

    def log_params(self, params: dict[str, Any]) -> None:
        import mlflow

        mlflow.log_params({k: v for k, v in params.items() if v is not None})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        import mlflow

        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

    def log_artifact(self, local_path: Path) -> None:
        import mlflow

        p = Path(local_path)
        if p.exists():
            mlflow.log_artifact(str(p))

    def log_model(self, model: object, artifact_path: str) -> None:
        """Log a fitted model using the appropriate MLflow flavor."""
        try:
            import sklearn.base

            if isinstance(model, sklearn.base.BaseEstimator):
                import mlflow.sklearn

                mlflow.sklearn.log_model(model, artifact_path)
                return
        except ImportError:
            pass
        # Keras / TF models: log the saved directory as a generic artifact.
        if hasattr(model, "save") and hasattr(model, "_model"):
            import mlflow

            mlflow.log_artifact(str(artifact_path))


def build_logger(
    name: str,
    experiment_name: str = "anomaliz",
    run_name: str | None = None,
    tracking_uri: str | None = None,
) -> NoOpLogger | MLflowLogger:
    """Factory: ``"mlflow"`` returns an MLflowLogger; anything else returns NoOpLogger."""
    if name == "mlflow":
        return MLflowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=tracking_uri,
        )
    return NoOpLogger()
