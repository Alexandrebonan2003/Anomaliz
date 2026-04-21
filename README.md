# Anomaliz

End-to-end anomaly detection service for multivariate time-series system metrics.
Combines classical ML, deep learning, a LangGraph explainability agent, MLflow experiment tracking, and a FastAPI serving layer.

For experimental methodology, model comparison results, and research findings, see [`README`](README).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Training CLI  python -m anomaliz.training.cli                  │
│                                                                  │
│  Data          Synthetic series → anomaly injection             │
│  Preprocessing Min-max normalisation → sliding windows          │
│  Models        IsolationForest  ·  LSTM Autoencoder             │
│                LSTM Forecaster  (best single model)             │
│  Scoring       IF + LSTM-AE weighted fusion → threshold         │
│  Tracking      MLflow (params · metrics · artifacts)            │
│  Bundle        artifacts/<run>/  ← normalizer + models +        │
│                                    thresholds + metrics.json    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ ANOMALIZ_ARTIFACT_DIR
┌──────────────────────────────▼──────────────────────────────────┐
│  FastAPI  POST /analyze                                          │
│                                                                  │
│  Detection     IsolationForest + LSTM Autoencoder fusion         │
│  Agent         LangGraph  →  analysis · severity · recommend    │
│  (optional)    OpenAI gpt-4o-mini  |  Ollama llama3.2           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick start

```bash
# 1. Install
make install

# 2. Train (writes bundle to artifacts/dev/)
make train

# 3. Serve
make serve
```

The API is now live at `http://localhost:8000`.

---

## Installation

Requires Python 3.11+.

```bash
make install
# or manually:
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

---

## Training

### Default run

```bash
make train                        # → artifacts/dev/
make train BUNDLE=artifacts/v1    # custom output path
```

### With MLflow tracking

```bash
make train-mlflow                 # logs to mlruns.db (SQLite)
make mlflow-ui                    # open http://localhost:5000
```

Or specify a remote tracking server:

```bash
.venv/bin/python -m anomaliz.training.cli \
  --out artifacts/v1 \
  --logger mlflow \
  --experiment my-experiment \
  --tracking-uri http://my-mlflow-server:5000
```

### With ablation sweep

```bash
make train-sweep    # runs capacity + window-size ablations
```

### Bundle contents

After training, `artifacts/dev/` contains:

```
artifacts/dev/
├── isolation_forest/   model.pkl + score_stats.json
├── lstm_autoencoder/   model.keras + params.json
├── lstm_forecaster/    model.keras + params.json
├── normalizer.json
├── threshold.json      per-detector + fusion thresholds
├── metadata.json       resolved config + git SHA + timestamp
└── metrics.json        F1 · precision · recall · ROC-AUC
                        ROC curve points · multi-seed aggregate
                        comparison_summary verdict
```

---

## API

### Start

```bash
make serve                        # uses artifacts/dev/
make serve BUNDLE=artifacts/v1    # custom bundle
```

### Endpoints

#### `GET /health`

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

#### `GET /metrics`

Returns `metrics.json` from the loaded bundle.

```bash
curl http://localhost:8000/metrics | python3 -m json.tool
```

#### `POST /analyze`

Send a window of 10 normalised values per feature:

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "cpu":     [0.28, 0.31, 0.29, 0.33, 0.30, 0.32, 0.28, 0.30, 0.31, 0.29],
    "memory":  [0.51, 0.52, 0.50, 0.51, 0.53, 0.50, 0.51, 0.52, 0.51, 0.50],
    "latency": [14.1, 15.0, 13.8, 14.5, 15.2, 14.0, 13.9, 14.7, 15.1, 14.3]
  }' | python3 -m json.tool
```

Example response (normal window, agent disabled):

```json
{
  "anomaly": false,
  "score": 0.112,
  "threshold": 0.130,
  "model_used": "isolation_forest",
  "analysis": null,
  "severity": null,
  "recommendation": null
}
```

Example response (anomalous window, agent enabled):

```json
{
  "anomaly": true,
  "score": 0.871,
  "threshold": 0.130,
  "model_used": "lstm_autoencoder",
  "analysis": "A CPU spike is detected alongside elevated latency, suggesting a resource-contended process.",
  "severity": "critical",
  "recommendation": "Identify the runaway process via top/htop and consider horizontal scaling."
}
```

`analysis`, `severity`, and `recommendation` are `null` when the agent is disabled or when no anomaly is detected.

---

## Explainability agent

The LangGraph agent enriches anomaly responses with structured interpretation. It is **disabled by default** — detection never depends on it.

### Enable with Ollama (local)

```bash
# Pull the model once
ollama pull llama3.2

make serve-with-agent   # sets ANOMALIZ__AGENT__BACKEND=ollama
```

### Enable with OpenAI

```bash
ANOMALIZ_ARTIFACT_DIR=artifacts/dev \
ANOMALIZ__AGENT__BACKEND=openai \
OPENAI_API_KEY=sk-... \
.venv/bin/uvicorn anomaliz.api.main:app
```

### Agent pipeline

```
analyze_node  →  severity_node  →  recommend_node
```

If the LLM call fails for any reason (network error, timeout, missing key), the API returns `200` with `null` agent fields — detection is never interrupted.

---

## Visualisation

Generate all performance plots from any trained bundle:

```bash
make dashboard                        # → reports/
make dashboard BUNDLE=artifacts/v1    # custom bundle
make dashboard REPORTS=my_reports     # custom output directory
```

| File | Contents |
|---|---|
| `roc_curves.png` | Overlaid ROC curves for all 4 detectors |
| `metrics_comparison.png` | Grouped bar chart: F1 / precision / recall |
| `seed_stability.png` | F1 mean ± std across 5 seeds |
| `comparison_summary.png` | Verdict table vs Isolation Forest baseline |

---

## Docker

### Start API + MLflow UI

```bash
make docker-up
```

- API: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`

Artifacts are mounted from `./artifacts` (read-only). Train a bundle locally first, then set `ANOMALIZ_ARTIFACT_DIR` in `.env` (copy `.env.example`).

### Also start Ollama

```bash
make docker-up-llm
```

Ollama is behind the `llm` compose profile so it is not pulled by default.

---

## Configuration

All hyperparameters live in `anomaliz/config/defaults.yaml` and can be overridden with env vars (`ANOMALIZ__<SECTION>__<KEY>`) or a YAML file (`--config`).

| Env var | Default | Purpose |
|---|---|---|
| `ANOMALIZ_ARTIFACT_DIR` | — | Bundle directory for the API |
| `ANOMALIZ__DATA__WINDOW_SIZE` | `10` | Sliding window length |
| `ANOMALIZ__DATA__N_POINTS` | `8000` | Series length for training |
| `ANOMALIZ__DETECTION__FUSION__WEIGHT_LSTM` | `0.7` | LSTM weight in fused score |
| `ANOMALIZ__AGENT__BACKEND` | `disabled` | `openai` · `ollama` · `disabled` |
| `ANOMALIZ__AGENT__OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `ANOMALIZ__AGENT__OLLAMA_MODEL` | `llama3.2` | Ollama model name |
| `ANOMALIZ__TRACKING__EXPERIMENT_NAME` | `anomaliz` | MLflow experiment name |
| `ANOMALIZ__TRACKING__TRACKING_URI` | `null` | MLflow tracking URI |
| `OPENAI_API_KEY` | — | Required when backend is `openai` |

---

## Tests

```bash
make test           # full suite (~60 s, includes training smoke test)
make test-fast      # skips training smoke test
```

---

## Project structure

```
anomaliz/
├── agent/          LangGraph explainability agent
│   ├── graph.py    compiled StateGraph (analyze → severity → recommend)
│   ├── llm.py      LLMBackend protocol + OpenAI / Ollama / Mock impls
│   ├── nodes.py    three prompt-driven node functions
│   └── state.py    AnomalyState TypedDict
├── api/            FastAPI application
│   ├── deps.py     Bundle + agent_graph dependency providers
│   ├── main.py     endpoints: /health /metrics /analyze
│   └── schemas.py  Pydantic request / response models
├── config/         Settings (pydantic-settings) + defaults.yaml
├── core/           Detector + ExperimentLogger protocols
├── data/           Synthetic series generator + train/val/test split
├── detection/      fuse() + decide() scoring utilities
├── models/         IFDetector · LSTMAutoencoder · LSTMForecaster
├── preprocessing/  MinMaxNormalizer + sliding window builder
├── tracking/       NoOpLogger + MLflowLogger + build_logger()
├── training/       pipeline.py (end-to-end run) + CLI
└── visualization/  dashboard.py (ROC curves · metrics · seed stability)
artifacts/          gitignored — output of training runs
tests/              pytest suite mirroring package layout
Dockerfile
docker-compose.yml
Makefile
```
