from __future__ import annotations

import json
from pathlib import Path

import pytest

from anomaliz.core.protocols import ExperimentLogger
from anomaliz.tracking.loggers import MLflowLogger, NoOpLogger, build_logger


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------

def test_noop_implements_protocol():
    assert isinstance(NoOpLogger(), ExperimentLogger)


def test_mlflow_implements_protocol():
    assert isinstance(MLflowLogger(), ExperimentLogger)


# ---------------------------------------------------------------------------
# NoOpLogger
# ---------------------------------------------------------------------------

def test_noop_logger_is_silent(tmp_path):
    logger = NoOpLogger()
    with logger:
        logger.log_params({"window_size": 10, "seed": 42})
        logger.log_metrics({"if_f1": 0.85, "fused_roc_auc": 0.91})
        logger.log_artifact(tmp_path / "nonexistent.json")
        logger.log_model(object(), "some_model")


def test_build_logger_noop_returns_noop():
    assert isinstance(build_logger("noop"), NoOpLogger)


def test_build_logger_unknown_falls_back_to_noop():
    assert isinstance(build_logger("unknown_backend"), NoOpLogger)


# ---------------------------------------------------------------------------
# MLflowLogger
# ---------------------------------------------------------------------------

def test_mlflow_logger_logs_params_and_metrics(tmp_path):
    import mlflow

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    with MLflowLogger(experiment_name="test_exp", tracking_uri=uri) as log:
        log.log_params({"window_size": 10, "seed": 42})
        log.log_metrics({"if_f1": 0.85, "fused_roc_auc": 0.91})

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["test_exp"])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["params.window_size"] == "10"
    assert row["params.seed"] == "42"
    assert float(row["metrics.if_f1"]) == pytest.approx(0.85)
    assert float(row["metrics.fused_roc_auc"]) == pytest.approx(0.91)


def test_mlflow_logger_skips_nonexistent_artifact(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    with MLflowLogger(tracking_uri=uri) as log:
        log.log_artifact(tmp_path / "does_not_exist.json")


def test_mlflow_logger_logs_sklearn_model(tmp_path):
    import mlflow
    from sklearn.ensemble import IsolationForest

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    model = IsolationForest(n_estimators=10).fit([[0.1, 0.2, 0.3]] * 20)
    with MLflowLogger(tracking_uri=uri) as log:
        log.log_model(model, "test_if_model")

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["anomaliz"])
    assert len(runs) == 1


def test_build_logger_mlflow_returns_mlflow_logger(tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    logger = build_logger("mlflow", tracking_uri=uri)
    assert isinstance(logger, MLflowLogger)


# ---------------------------------------------------------------------------
# ROC curves in metrics.json
# ---------------------------------------------------------------------------

def test_roc_curves_in_bundle_metrics(trained_bundle_dir):
    metrics = json.loads((trained_bundle_dir / "metrics.json").read_text())
    for detector in ("isolation_forest", "lstm_autoencoder", "lstm_forecaster", "fused"):
        m = metrics[detector]
        rc = m.get("roc_curve")
        assert rc is not None, f"roc_curve missing for {detector}"
        assert isinstance(rc["fpr"], list) and len(rc["fpr"]) > 1
        assert isinstance(rc["tpr"], list)
        assert len(rc["fpr"]) == len(rc["tpr"])
        assert rc["fpr"][0] == pytest.approx(0.0, abs=0.01)
        assert rc["tpr"][-1] == pytest.approx(1.0, abs=0.01)


def test_roc_curve_values_in_unit_interval(trained_bundle_dir):
    metrics = json.loads((trained_bundle_dir / "metrics.json").read_text())
    for detector in ("isolation_forest", "lstm_forecaster"):
        rc = metrics[detector]["roc_curve"]
        assert all(0.0 <= v <= 1.0 for v in rc["fpr"])
        assert all(0.0 <= v <= 1.0 for v in rc["tpr"])


# ---------------------------------------------------------------------------
# End-to-end: pipeline logs to MLflow
# ---------------------------------------------------------------------------

def test_pipeline_mlflow_run_is_recorded(tmp_path, small_settings):
    import mlflow

    from anomaliz.training.pipeline import run_training

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    bundle_dir = tmp_path / "bundle"
    logger = MLflowLogger(experiment_name="test_pipeline", tracking_uri=uri)
    run_training(small_settings, bundle_dir, logger=logger)

    mlflow.set_tracking_uri(uri)
    runs = mlflow.search_runs(experiment_names=["test_pipeline"])
    assert len(runs) == 1
    row = runs.iloc[0]
    # Key params logged
    assert row["params.seed"] == str(small_settings.seed)
    assert row["params.window_size"] == str(small_settings.data.window_size)
    # Key metrics logged with correct prefix
    assert float(row["metrics.if_f1"]) >= 0.0
    assert float(row["metrics.fused_f1"]) >= 0.0
