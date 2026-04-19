# AI-Powered Anomaly Detection System

## Project Overview

Python project for anomaly detection on time-series system metrics (CPU, memory, latency).
Combines classical ML, deep learning, and a LangGraph agent for explainability.
Exposed via a FastAPI REST service. Experiment tracking with MLflow.

**Tech stack:** Python 3.11, scikit-learn, TensorFlow/Keras, LangGraph, LangChain, FastAPI, Pydantic, MLflow, pytest, Docker

---

## Project Structure

```
anomaly_detection/
├── data/
│   ├── generator.py          # Synthetic time-series data generation
│   └── dataset.py            # Train/val/test split, label handling
├── preprocessing/
│   └── normalizer.py         # Min-max normalization + sliding window builder
├── models/
│   ├── isolation_forest.py   # sklearn IsolationForest wrapper + scoring
│   └── lstm_autoencoder.py   # Keras LSTM Autoencoder + training + reconstruction scoring
├── detection/
│   └── scorer.py             # Score fusion (IF + LSTM), thresholding, binary decision
├── agent/
│   ├── state.py              # AnomalyState TypedDict (LangGraph shared state)
│   ├── nodes.py              # analyze_node, severity_node, recommend_node
│   └── graph.py              # LangGraph graph construction and compilation
├── api/
│   ├── main.py               # FastAPI application
│   └── schemas.py            # Pydantic input/output models
├── visualization/
│   └── dashboard.py          # Time-series plots, anomaly highlights, IF vs LSTM comparison
├── tracking/
│   └── mlflow_logger.py      # MLflow logging wrapper
├── notebooks/
│   └── exploration.ipynb     # EDA, data visualization, model debugging
├── tests/
│   ├── test_generator.py
│   ├── test_models.py
│   └── test_api.py           # FastAPI tests with httpx TestClient
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Data

### Synthetic dataset schema

```
timestamp | cpu (float [0,1]) | memory (float [0,1]) | latency (float, ms) | label (int {0,1})
```

### Normal behavior

- CPU: 20–40%, gaussian noise (σ ≈ 0.05)
- Memory: 45–55%, stable
- Latency: 10–20 ms

### Injected anomaly types

| Type            | Behavior                                             | Duration     |
|-----------------|------------------------------------------------------|--------------|
| `cpu_spike`     | Sudden spike to ~90–95%                              | 2–5 points   |
| `memory_leak`   | Slow monotonic rise, does not recover                | 20–50 points |
| `latency_drift` | Slow drift upward over a time window                 | 10–30 points |
| `system_crash`  | All metrics drop to ~0 simultaneously                | 3–8 points   |

Injection is parameterizable: probability, intensity, duration.

---

## Feature Engineering

### Sliding window

```python
window_size = 10  # hyperparameter

# For LSTM Autoencoder:
X.shape = (batch_size, 10, 3)   # (batch, timesteps, features)

# For Isolation Forest:
X.shape = (batch_size, 30)      # flattened window
```

---

## Models

### Isolation Forest

```python
from sklearn.ensemble import IsolationForest

IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
```

- Unsupervised, trained on raw data (normal + anomalies)
- Returns continuous anomaly score (more negative = more anomalous)
- Normalize output to [0, 1] for fusion

### LSTM Autoencoder

```
Encoder:
  Input: (batch, 10, 3)
  LSTM(64, return_sequences=True) → Dropout(0.2)
  LSTM(32, return_sequences=False) → latent vector (dim=32)

Decoder:
  RepeatVector(10)
  LSTM(32, return_sequences=True) → Dropout(0.2)
  LSTM(64, return_sequences=True)
  TimeDistributed(Dense(3)) → reconstructed X̂

Loss: MSE
Optimizer: Adam(lr=1e-3)
```

- Trained on **normal data only**
- Anomaly = high reconstruction error
- Threshold: `mean(val_errors) + k * std(val_errors)`, k ≈ 2.5 (tunable)

---

## Anomaly Scoring

```python
# Per-window reconstruction error
reconstruction_error = MAE(X, X_reconstructed)   # shape: (batch,)

# Normalize to [0, 1]
score_lstm = clip((error - min_err) / (max_err - min_err), 0, 1)

# Weighted fusion with IF score
final_score = 0.3 * score_if + 0.7 * score_lstm

# Decision
anomaly: bool = final_score > threshold  # threshold ≈ 0.5–0.7
```

---

## LangGraph Agent

Triggered only when `anomaly == True`. Enriches the raw score with structured analysis.

### Shared state

```python
class AnomalyState(TypedDict):
    cpu: float
    memory: float
    latency: float
    score: float
    threshold: float
    model_used: str
    analysis: str       # filled by analyze_node
    severity: str       # filled by severity_node — values: "low" | "medium" | "critical"
    recommendation: str # filled by recommend_node
```

### Graph

```
START → analyze_node → severity_node → recommend_node → END
```

### Node prompts

**analyze_node:**
```
You are a senior SRE. Given these system metrics at time T:
CPU: {cpu}%, Memory: {memory}%, Latency: {latency}ms
Anomaly score: {score:.2f} (threshold: {threshold:.2f}), detected by: {model_used}
In 2 sentences, identify the probable anomaly type and its likely cause.
```

**severity_node:**
```
Given this analysis: {analysis}
And anomaly score: {score:.2f}
Classify severity as exactly one of: low / medium / critical
Reply with the keyword only.
```

**recommend_node:**
```
Anomaly severity: {severity}. Analysis: {analysis}
Generate a concrete 1–2 sentence action recommendation for a DevOps engineer.
```

### LLM backend

- Default: OpenAI API (`gpt-4o-mini`) — configure via `OPENAI_API_KEY` env var
- Alternative: Ollama local (`llama3.2` or `mistral`) — set `LLM_BACKEND=ollama` in config

---

## API (FastAPI)

### POST /analyze

**Input schema:**
```json
{
  "cpu":     [float, ...],      // 10 values, normalized [0,1]
  "memory":  [float, ...],      // 10 values, normalized [0,1]
  "latency": [float, ...]       // 10 values, in ms
}
```

**Output schema:**
```json
{
  "anomaly": true,
  "score": 0.87,
  "threshold": 0.65,
  "model_used": "lstm_autoencoder",
  "analysis": "...",
  "severity": "critical",
  "recommendation": "..."
}
```

### GET /health

Returns `{"status": "ok"}`.

### GET /metrics

Returns latest evaluation metrics: F1, ROC AUC, precision, recall.

---

## MLflow Tracking

```python
with mlflow.start_run():
    mlflow.log_param("window_size", 10)
    mlflow.log_param("lstm_units", [64, 32])
    mlflow.log_param("threshold_k", 2.5)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.sklearn.log_model(isolation_forest, "isolation_forest")
    mlflow.keras.log_model(lstm_autoencoder, "lstm_autoencoder")
```

Run `mlflow ui` locally to inspect experiments.

---

## Evaluation

Labels from injected anomalies provide ground truth for supervised evaluation of the unsupervised pipeline.

Metrics to compute: **F1-score** (primary), ROC AUC, Precision, Recall.
Evaluate IF and LSTM Autoencoder separately, then compare.

---

## Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```