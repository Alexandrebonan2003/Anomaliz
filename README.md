# Anomaliz

End-to-end anomaly detection service for multivariate time-series system metrics.
Combines classical ML, deep learning, a LangGraph explainability agent, MLflow experiment tracking, and a FastAPI serving layer.

For experimental methodology, model comparison results, and research findings, see [`README_research.md`](README_research.md).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Training CLI  python -m anomaliz.training.cli                  │
│                                                                  │
│  Data          Synthetic series → anomaly injection             │
│  Preprocessing Min-max normalisation → sliding windows          │
│  Models        IsolationForest  ·  LSTM Autoencoder             │
│                LSTM Forecaster  (best synthetic single model)   │
│  Scoring       IF + LSTM-AE weighted fusion → threshold         │
│  Tracking      MLflow (params · metrics · artifacts)            │
│  Bundle        artifacts/<run>/  ← normalizer + models +        │
│                                    thresholds + metrics.json    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ ANOMALIZ_ARTIFACT_DIR
┌──────────────────────────────▼──────────────────────────────────┐
│  FastAPI  POST /analyze                                          │
│                                                                  │
│  Detection     score + threshold + selected detector             │
│  Agent         LangGraph  →  analysis · severity · recommend    │
│  (optional)    OpenAI gpt-4o-mini  |  Ollama llama3.2           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick start

```bash
make install
make train
make serve
```

The API is live at `http://localhost:8000`.

---

## End-to-end workflow

Typical local workflow:

```bash
make install
make train
make serve
```

Optional extensions:

```bash
make train-mlflow
make mlflow-ui
make dashboard
make eval-nab
make docker-up
```

This covers training, external validation, serving, visualisation, and deployment.

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
make train
make train BUNDLE=artifacts/v1
```

### With MLflow tracking

```bash
make train-mlflow
make mlflow-ui
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
make train-sweep
```

### Bundle contents

After training, `artifacts/dev/` contains:

```
artifacts/dev/
├── isolation_forest/   model.pkl + score_stats.json
├── lstm_autoencoder/   model.keras + params.json
├── lstm_forecaster/    model.keras + params.json
├── normalizer.json
├── threshold.json
├── metadata.json
└── metrics.json
```

---

## API

### Start

```bash
make serve
make serve BUNDLE=artifacts/v1
```

### Endpoints

#### `GET /health`

```bash
curl http://localhost:8000/health
```

#### `GET /metrics`

```bash
curl http://localhost:8000/metrics | python3 -m json.tool
```

#### `POST /analyze`

```bash
curl -s -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "cpu":     [0.28, 0.31, 0.29, 0.33, 0.30, 0.32, 0.28, 0.30, 0.31, 0.29],
    "memory":  [0.51, 0.52, 0.50, 0.51, 0.53, 0.50, 0.51, 0.52, 0.51, 0.50],
    "latency": [14.1, 15.0, 13.8, 14.5, 15.2, 14.0, 13.9, 14.7, 15.1, 14.3]
  }' | python3 -m json.tool
```

`analysis`, `severity`, and `recommendation` are `null` when the agent is disabled or when no anomaly is detected.

---

## Explainability agent

The LangGraph agent enriches anomaly responses with structured interpretation. It is **disabled by default**.

### Enable with Ollama

```bash
ollama pull llama3.2
make serve-with-agent
```

### Enable with OpenAI

```bash
ANOMALIZ_ARTIFACT_DIR=artifacts/dev \
ANOMALIZ__AGENT__BACKEND=openai \
OPENAI_API_KEY=sk-... \
.venv/bin/uvicorn anomaliz.api.main:app
```

If the LLM call fails for any reason, the API returns `200` with `null` agent fields — detection is never interrupted.

---

## Visualisation

Generate all performance plots from any trained bundle:

```bash
make dashboard
make dashboard BUNDLE=artifacts/v1
make dashboard REPORTS=my_reports
```

| File | Contents |
|---|---|
| `roc_curves.png` | Overlaid ROC curves for all 4 detectors |
| `metrics_comparison.png` | Grouped bar chart: F1 / precision / recall |
| `seed_stability.png` | F1 mean ± std across 5 seeds |
| `comparison_summary.png` | Verdict table vs Isolation Forest baseline |

---

## External evaluation (NAB)

Run external validation on real-world time series:

```bash
make eval-nab
```

Options:

```bash
make eval-nab BUNDLE=artifacts/v1
python -m anomaliz.data.nab --bundle artifacts/dev --series cpu_asg
```

This step:
- downloads and caches NAB datasets locally (`.nab_cache/`)
- adapts univariate series to the project’s multivariate format
- reuses normalisation, windowing, and threshold tuning
- reports metrics comparable to the synthetic benchmark

---

## Docker

### Start API + MLflow UI

```bash
make docker-up
```

- API: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`

Artifacts are mounted from `./artifacts` (read-only). Train a bundle locally first, then set `ANOMALIZ_ARTIFACT_DIR` in `.env`.

### Also start Ollama

```bash
make docker-up-llm
```

Ollama is behind the `llm` compose profile so it is not pulled by default.

This setup reproduces the full system (API, tracking, and optional LLM agent) in a portable and reproducible environment.

---

## Configuration

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
make test
make test-fast
```

---

## Project structure

```
anomaliz/
├── agent/
├── api/
├── config/
├── core/
├── data/
├── detection/
├── models/
├── preprocessing/
├── tracking/
├── training/
├── visualization/
artifacts/
tests/
Dockerfile
docker-compose.yml
Makefile
```
