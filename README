# Anomaliz — Time Series Anomaly Detection: Research Notes

## Overview

Anomaliz is an end-to-end anomaly detection system for multivariate time-series system metrics (CPU, memory, latency). The project is structured around five phases, progressing from a classical ML baseline to a production-ready service with deep learning, LLM-assisted interpretation, experiment tracking, and containerised deployment.

The research focus is on:

- rigorous evaluation methodology and threshold calibration
- comparison of anomaly detection formulations (isolation, reconstruction, forecasting)
- realistic synthetic benchmark design
- multi-seed statistical validation

For installation, API usage, Docker, and configuration, see [`README.md`](README.md).

---

## System Components

| Phase | Component | Description |
|---|---|---|
| 1 | Isolation Forest | Classical ML baseline on flattened sliding windows |
| 2 | LSTM Autoencoder | Reconstruction-based anomaly scoring (normal-only training) |
| 2.5 | LSTM Forecaster | Forecasting-based anomaly scoring; best single model |
| 2 | Score fusion | Weighted combination of IF and LSTM-AE scores |
| 3 | LangGraph agent | Structured LLM interpretation: analysis, severity, recommendation |
| 4 | MLflow tracking | Parameter, metric, and artifact logging per training run |
| 4 | Evaluation dashboard | ROC curves, metric comparison, seed-stability plots |
| 5 | FastAPI + Docker | REST serving layer with containerised deployment |

---

## Pipeline

### Global workflow

```text
Raw multivariate time series
   ↓
Synthetic anomaly injection
   ↓
Sliding windows
   ↓
Window labeling (last-point strategy)
   ↓
Model-specific training / scoring
   ↓
Threshold tuning on validation
   ↓
Final evaluation on test
   ↓
Artifact bundle (models + thresholds + metrics + metadata)
```

### Isolation Forest (Phase 1)

```text
Windows → flattening → IsolationForest → anomaly scores
→ threshold tuning → final evaluation
```

### Reconstruction-based DL (Phase 2)

```text
Windows → normal-only subset → LSTM Autoencoder
→ reconstruction error → threshold tuning → final evaluation
```

### Forecasting-based DL (Phase 2.5)

```text
Windows → normal-only subset → LSTM Forecaster
→ forecast residuals → threshold tuning → final evaluation
```

---

## Data and Modeling Insights

### 1. Window labeling strategy matters

The initial setup labeled a window as anomalous if **any point inside the window** was anomalous. This created an unrealistic anomaly rate and artificially inflated results. After switching to a **last-point labeling strategy** (`label = last element of the window`), the benchmark became more realistic and the task more meaningful.

This confirmed that dataset construction has a major impact on measured anomaly detection performance.

### 2. Threshold calibration is a core part of anomaly detection

The pipeline enforces a strict separation between:
- model fitting
- threshold tuning on validation
- final evaluation on test

This avoids conflating ranking quality with decision calibration and makes the reported results more defensible. It also showed that anomaly detection performance is often constrained more by **data design and threshold choice** than by the model itself.

### 3. Realistic anomaly rates are critical

A major part of the project was making the synthetic benchmark both realistic enough to resemble rare-event detection and stable enough to support multi-seed evaluation. The final setup keeps the anomaly rate low after windowing while ensuring valid splits across seeds.

Typical class balance in the final benchmark:

| Split | Anomaly rate |
|---|---:|
| Train | ~10.1% |
| Validation | ~9.7% |
| Test | ~6.8% |

This preserves an imbalanced anomaly-detection setting while keeping evaluation statistically meaningful.

---

## Experimental Results

### Phase 2 — Initial DL transition

On the first realistic benchmark, the LSTM Autoencoder appeared to improve over the Isolation Forest baseline:

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Isolation Forest | ~0.43 | ~0.87 | ~0.57 | ~0.93 |
| LSTM Autoencoder | ~0.50 | ~0.90 | ~0.64 | ~0.93 |
| Fused | ~0.48 | ~0.90 | ~0.63 | ~0.93–0.94 |

This validated the move toward sequence modeling, but only on a single-seed setting.

### Phase 2.5 — Multi-seed robust evaluation

Final aggregated results across **5 seeds** on the stabilised realistic benchmark:

| Model | F1 mean ± std | Precision mean ± std | Recall mean ± std | ROC-AUC mean ± std | Verdict |
|---|---:|---:|---:|---:|---|
| Isolation Forest | 0.642 ± 0.067 | 0.596 ± 0.096 | 0.702 ± 0.037 | 0.904 ± 0.024 | baseline |
| LSTM Autoencoder | 0.655 ± 0.048 | 0.590 ± 0.076 | 0.744 ± 0.025 | 0.908 ± 0.027 | no improvement |
| Fused (IF + AE) | 0.663 ± 0.049 | 0.597 ± 0.077 | 0.754 ± 0.026 | 0.910 ± 0.027 | no improvement |
| **LSTM Forecaster** | **0.797 ± 0.056** | **0.897 ± 0.113** | **0.723 ± 0.038** | **0.936 ± 0.007** | **beats baseline** |

### Interpretation

- The **LSTM Autoencoder** does not robustly outperform the Isolation Forest baseline. The single-seed gain seen in Phase 2 does not hold under multi-seed evaluation.
- Score fusion over IF + AE provides no decisive improvement.
- The **LSTM Forecaster** clearly outperforms both the ML baseline and the reconstruction-based DL approach.

This suggests that, on this benchmark, **forecasting is a more effective anomaly detection formulation than reconstruction**. The forecaster is therefore the primary model used at inference time.

---

## Ablation Insights

Ablations were run over latent/model capacity (`units_2`) and window size.

- Changing Autoencoder capacity did not materially alter the global conclusion.
- The Forecaster remained the strongest DL model across all tested settings.
- With larger windows, classical and reconstruction-based approaches degraded more sharply, while the Forecaster remained robust.

This indicates that the forecasting formulation makes better use of temporal context on this synthetic benchmark.

---

## Key Takeaways

- Dataset construction strongly impacts anomaly detection performance.
- Window labeling strategy can artificially inflate results.
- Threshold calibration is essential — high ROC-AUC does not guarantee strong thresholded classification.
- A deep learning model does not automatically outperform a well-tuned classical baseline.
- On this benchmark, **forecasting-based DL outperforms reconstruction-based DL**.
- Robust multi-seed evaluation is necessary to validate performance claims.

---

## Technical Highlights

- End-to-end anomaly detection pipeline with training/serving separation
- Synthetic time-series benchmark engineering with realistic anomaly injection
- Isolation Forest, LSTM Autoencoder, and LSTM Forecaster implementations
- Reconstruction-based vs forecasting-based anomaly detection comparison
- Validation-based threshold optimisation
- Multi-seed evaluation with metric aggregation
- MLflow experiment tracking (params, metrics, artifacts)
- LangGraph agent for structured LLM-assisted anomaly interpretation
- Performance dashboard: ROC curves, metric comparison, seed-stability plots
- Reproducible artifact bundles for training/serving separation
- REST API via FastAPI; containerised deployment via Docker
