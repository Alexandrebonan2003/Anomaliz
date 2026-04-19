# Anomaliz — Implementation Plan

## Context

The repository at [/home/alex/Anomaliz/](/home/alex/Anomaliz/) currently contains only [ADS_system.md](../../Anomaliz/ADS_system.md), `.gitignore`, `.git`, and the newly-added [CLAUDE.md](../../Anomaliz/CLAUDE.md). The spec describes an anomaly detection system combining scikit-learn IsolationForest, a Keras LSTM Autoencoder, a LangGraph agent, FastAPI serving, and MLflow tracking.

This plan delivers a **strict MVP-first** build:

- Phase 1 is a true minimal end-to-end path: one detector, one pipeline, one bundle, one API — fast to implement, easy to debug, low in dependency weight.
- Additional detectors, fusion, the LLM agent, tracking, Docker, and dashboards are layered on in distinct later phases.
- The architectural invariants (config-driven hyperparameters, training/serving separation via artifact bundle, `Detector` protocol) are established in Phase 1 so that later phases add code without reshaping the system.

Confirmed scoping decisions:
- Package name: `anomaliz`.
- Phase 1 ships **IsolationForest only** — LSTM Autoencoder and fusion move to Phase 2.
- LangGraph agent → Phase 3.
- MLflow → Phase 4.

---

## Target package layout

Full target layout. Directories are added phase by phase — don't pre-create empty packages.

```
anomaliz/
  config/
    __init__.py
    settings.py           # Settings class, YAML loader
    defaults.yaml         # hyperparameters
  core/
    __init__.py
    protocols.py          # Detector (Phase 1); LLMBackend, ExperimentLogger added later
  data/
    generator.py          # synthetic series + anomaly injection
    dataset.py            # split
  preprocessing/
    normalizer.py         # MinMaxNormalizer (fit/transform/save/load)
    windowing.py          # make_windows, flatten_windows, window_labels
  models/
    isolation_forest.py   # IFDetector (Phase 1)
    lstm_autoencoder.py   # LSTMAutoencoder (Phase 2)
  detection/
    scorer.py             # decide() (Phase 1); fuse() added in Phase 2
    threshold.py          # threshold utilities
  agent/                  # Phase 3
  api/
    main.py               # FastAPI app, lifespan loads bundle
    schemas.py            # AnalyzeRequest, AnalyzeResponse
    deps.py               # get_settings, get_bundle, get_metrics
  tracking/               # Phase 4
  visualization/          # Phase 5
  training/
    pipeline.py           # orchestrates training run
    cli.py                # python -m anomaliz.training.cli
artifacts/                # gitignored, output of training runs
tests/
  test_generator.py
  test_preprocessing.py
  test_if_detector.py
  test_training_smoke.py
  test_api.py
pyproject.toml
Dockerfile                # Phase 5
```

**Additions beyond the spec layout, and why:**

- `config/` — enforces the "no hardcoded hyperparameters" invariant.
- `core/protocols.py` — single home for real abstractions.
- `training/` — separates offline model-building from online serving.
- `artifacts/` — the handoff contract between training and the API.
- `pyproject.toml` replaces `requirements.txt`.

---

## Core interfaces

Only abstract where there is a clear second implementation or test-double need. Introduce each protocol in the phase where it earns its keep.

- **Phase 1:** `Detector` (Protocol) — `fit(X)`, `score(X) -> np.ndarray[float in 0,1]`, `save(path)`, `load(path)`. Justified even with one implementation because it defines the contract Phase 2's LSTM and the Phase 1 serving layer both depend on.
- **Phase 2:** add a second `Detector` impl (`LSTMAutoencoder`); `fuse()` appears as a plain function, not a protocol.
- **Phase 3:** `LLMBackend` (Protocol) — three real impls (OpenAI, Ollama, mock).
- **Phase 4:** `ExperimentLogger` (Protocol) — NoOp vs. MLflow.

**Deliberately not abstracted:** Normalizer, windowing, fusion, thresholder. One implementation each; direct use.

---

## Configuration strategy

- `pydantic-settings` `Settings` class in [config/settings.py](../../Anomaliz/anomaliz/config/settings.py). Phase 1 sub-models: `DataConfig`, `ModelConfig` (only `IFConfig` at first), `DetectionConfig`, `APIConfig`. Add `AgentConfig` and `TrackingConfig` in the phases that introduce them — don't scaffold speculatively.
- Defaults live in [config/defaults.yaml](../../Anomaliz/anomaliz/config/defaults.yaml).
- Resolution order: YAML defaults → env var overrides (`ANOMALIZ__DATA__WINDOW_SIZE=...`) → optional `--config` file for the training CLI.
- A single `Settings` instance is built at process start and passed explicitly into constructors. Global singletons only where FastAPI's DI system needs them.
- The resolved config is snapshotted into `metadata.json` inside every artifact bundle.

---

## Training / inference separation

**Training CLI:** `python -m anomaliz.training.cli --config path.yaml --out artifacts/run_<ts>/`

**Phase 1 pipeline stages (in [training/pipeline.py](../../Anomaliz/anomaliz/training/pipeline.py)):**
1. Generate synthetic series with labels (seeded RNG).
2. Train/val/test split.
3. Fit `MinMaxNormalizer` on training data.
4. Build flattened sliding windows.
5. Fit `IFDetector` on training windows; the detector internally captures the score-normalization stats it needs to map `decision_function` output into `[0,1]`.
6. Evaluate on test set (F1, ROC AUC, precision, recall).
7. Persist bundle.

**Phase 1 bundle contents (`artifacts/<run>/`):**
- `isolation_forest/` — detector artifacts written by `IFDetector.save()` (pickle + any normalization stats).
- `normalizer.json`
- `threshold.json` (`decision_threshold`)
- `metadata.json` (resolved config, git SHA, timestamp)
- `metrics.json` (served verbatim by `/metrics`)

Phase 2 extends the bundle with `lstm_autoencoder/` and adds LSTM-specific threshold stats.

**Inference:** the API reads `ANOMALIZ_ARTIFACT_DIR` at startup (FastAPI lifespan), loads the bundle once into app state. No training code is imported at serve time.

---

## Testability strategy

- All randomness via explicit `np.random.default_rng(seed)`. No global RNG.
- Per-layer unit tests with small fixtures — never assert model convergence.
- `test_training_smoke.py` runs the full pipeline on a ~200-point series to produce a real bundle; `test_api.py` reuses that bundle via a pytest fixture (no checked-in binary artifact needed in Phase 1).
- API tested via FastAPI `TestClient`; `get_bundle` overridden with the fixture-trained bundle.

Phase 3 will add a `MockLLMBackend` so agent tests never call a real LLM.

---

## Phasing

### Phase 1 — MVP: end-to-end detection with IsolationForest only

Deliverables:
- Config system (minimal nesting), `Detector` protocol, data generator, preprocessing, `IFDetector`, threshold + decide.
- Training CLI producing a persisted bundle and `metrics.json`.
- FastAPI `/health`, `/analyze` (returns `anomaly`, `score`, `threshold`, `model_used="isolation_forest"`, agent fields null), `/metrics`.
- Full pytest suite green with seeds locked.
- `pyproject.toml` runtime deps: numpy, pandas, scikit-learn, fastapi, uvicorn, pydantic, pydantic-settings, pyyaml, pytest, httpx. **Notably absent: tensorflow, langgraph, mlflow.**

### Phase 2 — LSTM Autoencoder + score fusion

- Add `tensorflow` dependency.
- `LSTMAutoencoder` implementing `Detector`, trained on normal-only windows.
- `fuse()` in `detection/scorer.py` combining IF + LSTM scores.
- Training pipeline extended: step 5a fit LSTM, step 5b derive LSTM threshold from val errors, step 6 evaluate each detector and the fused scorer.
- `/analyze` now selects `model_used` based on which detector contributed more.
- Bundle gains `lstm_autoencoder/` and LSTM-specific threshold stats.

### Phase 3 — LangGraph agent

- Add `langgraph`, `langchain-openai` deps.
- `LLMBackend` protocol + OpenAI, Ollama, Mock impls in `agent/llm.py`.
- `agent/state.py`, `nodes.py`, `graph.py` per spec.
- `/analyze` enriched with `analysis`, `severity`, `recommendation` when `anomaly=True`. Agent skipped when `anomaly=False`.

### Phase 4 — MLflow + evaluation polish

- `ExperimentLogger` protocol + `NoOpLogger` + `MLflowLogger`.
- Training CLI accepts `--logger mlflow`.
- Per-model (IF vs LSTM) breakdown, ROC curves in `metrics.json`.

### Phase 5 — Docker + dashboard

- `Dockerfile`, `docker-compose.yml` (API + MLflow UI + Ollama).
- `visualization/dashboard.py` (plotly) for offline run inspection.
- README.

---

## Critical files (Phase 1)

- [anomaliz/config/settings.py](../../Anomaliz/anomaliz/config/settings.py) — Settings + YAML loader.
- [anomaliz/config/defaults.yaml](../../Anomaliz/anomaliz/config/defaults.yaml) — Phase 1 hyperparameters only (window_size, IF params, decision_threshold, seed).
- [anomaliz/core/protocols.py](../../Anomaliz/anomaliz/core/protocols.py) — `Detector`.
- [anomaliz/data/generator.py](../../Anomaliz/anomaliz/data/generator.py) — seeded synthetic series with the 4 anomaly types.
- [anomaliz/preprocessing/normalizer.py](../../Anomaliz/anomaliz/preprocessing/normalizer.py) and [windowing.py](../../Anomaliz/anomaliz/preprocessing/windowing.py).
- [anomaliz/models/isolation_forest.py](../../Anomaliz/anomaliz/models/isolation_forest.py) — `IFDetector` implementing `Detector`.
- [anomaliz/detection/scorer.py](../../Anomaliz/anomaliz/detection/scorer.py) — `decide(score, threshold)`.
- [anomaliz/training/pipeline.py](../../Anomaliz/anomaliz/training/pipeline.py) and [cli.py](../../Anomaliz/anomaliz/training/cli.py).
- [anomaliz/api/main.py](../../Anomaliz/anomaliz/api/main.py), [schemas.py](../../Anomaliz/anomaliz/api/schemas.py), [deps.py](../../Anomaliz/anomaliz/api/deps.py).
- [pyproject.toml](../../Anomaliz/pyproject.toml), [.gitignore](../../Anomaliz/.gitignore) (add `artifacts/`, `mlruns/`, `__pycache__/`, `.venv/`).

---

## Verification (end of Phase 1)

- `pytest` — suite green, including `test_training_smoke.py` end-to-end on a ~200-point series.
- `python -m anomaliz.training.cli --out artifacts/dev/` — produces a complete bundle with non-empty `metrics.json`.
- `ANOMALIZ_ARTIFACT_DIR=artifacts/dev uvicorn anomaliz.api.main:app` — serves successfully.
- `curl localhost:8000/health` → `{"status":"ok"}`.
- `curl -X POST localhost:8000/analyze -d '{"cpu":[...10...],"memory":[...10...],"latency":[...10...]}'` → JSON with `anomaly`, `score`, `threshold`, `model_used="isolation_forest"`, agent fields null.
- `curl localhost:8000/metrics` → contents of `metrics.json`.
- Override a hyperparameter via env var (e.g. `ANOMALIZ__DATA__WINDOW_SIZE=15`) and confirm it propagates into the bundle's `metadata.json`.
