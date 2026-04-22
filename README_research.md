# Anomaliz — Time Series Anomaly Detection: Research Notes

## Overview

Anomaliz is an end-to-end anomaly detection system for multivariate time-series system metrics (CPU, memory, latency). The project progresses from a classical ML baseline to a complete service with deep learning, LLM-assisted interpretation, experiment tracking, external validation, and containerised deployment.

The research focus is on:
- rigorous evaluation methodology and threshold calibration
- comparison of anomaly detection formulations (isolation, reconstruction, forecasting)
- realistic synthetic benchmark design
- multi-seed statistical validation
- external validation on real-world data

For installation, API usage, Docker, and configuration, see [`README.md`](README.md).

---

## System Components

| Phase | Component | Description |
|---|---|---|
| 1 | Isolation Forest | Classical ML baseline on flattened sliding windows |
| 2 | LSTM Autoencoder | Reconstruction-based anomaly scoring (normal-only training) |
| 2.5 | LSTM Forecaster | Forecasting-based anomaly scoring; best single model on synthetic data |
| 2 | Score fusion | Weighted combination of IF and LSTM-AE scores |
| 3 | LangGraph agent | Structured LLM interpretation: analysis, severity, recommendation |
| 4 | MLflow tracking | Parameter, metric, and artifact logging per training run |
| 5 | Evaluation dashboard | ROC curves, metric comparison, seed-stability plots |
| 5 | FastAPI + Docker | REST serving layer with containerised deployment |
| 5 | NAB validation | External validation on real-world anomaly benchmark |

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

### 2. Threshold calibration is a core part of anomaly detection

The pipeline enforces a strict separation between:
- model fitting
- threshold tuning on validation
- final evaluation on test

This avoids conflating ranking quality with decision calibration and makes the reported results more defensible.

### 3. Realistic anomaly rates are critical

Typical class balance in the final synthetic benchmark:

| Split | Anomaly rate |
|---|---:|
| Train | ~10.1% |
| Validation | ~9.7% |
| Test | ~6.8% |

This preserves an imbalanced anomaly-detection setting while keeping evaluation statistically meaningful.

---

## Experimental Results

### Phase 2 — Initial DL transition

| Model | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|
| Isolation Forest | ~0.43 | ~0.87 | ~0.57 | ~0.93 |
| LSTM Autoencoder | ~0.50 | ~0.90 | ~0.64 | ~0.93 |
| Fused | ~0.48 | ~0.90 | ~0.63 | ~0.93–0.94 |

### Phase 2.5 — Multi-seed robust evaluation

Final aggregated results across **5 seeds** on the stabilised realistic benchmark:

| Model | F1 mean ± std | Precision mean ± std | Recall mean ± std | ROC-AUC mean ± std | Verdict |
|---|---:|---:|---:|---:|---|
| Isolation Forest | 0.642 ± 0.067 | 0.596 ± 0.096 | 0.702 ± 0.037 | 0.904 ± 0.024 | baseline |
| LSTM Autoencoder | 0.655 ± 0.048 | 0.590 ± 0.076 | 0.744 ± 0.025 | 0.908 ± 0.027 | no improvement |
| Fused (IF + AE) | 0.663 ± 0.049 | 0.597 ± 0.077 | 0.754 ± 0.026 | 0.910 ± 0.027 | no improvement |
| **LSTM Forecaster** | **0.797 ± 0.056** | **0.897 ± 0.113** | **0.723 ± 0.038** | **0.936 ± 0.007** | **beats baseline** |

### Interpretation

- The **LSTM Autoencoder** does not robustly outperform the Isolation Forest baseline.
- Score fusion over IF + AE provides no decisive improvement.
- The **LSTM Forecaster** clearly outperforms both the ML baseline and the reconstruction-based DL approach on the synthetic benchmark.

This suggests that, on the synthetic benchmark, **forecasting is a more effective anomaly detection formulation than reconstruction**.

---

## External Validation (NAB Benchmark)

To assess generalisation beyond synthetic data, the models were evaluated on real-world time series from the Numenta Anomaly Benchmark (NAB).

Two representative series were selected:
- `cpu_asg` (AWS Auto Scaling misconfiguration — clear distribution shift)
- `ec2_cpu_ac20cd` (long sustained anomaly period)

### Results

| Dataset / detector view | F1 | ROC-AUC |
|---|---:|---:|
| Synthetic (fused, multi-seed) | ~0.64 | ~0.90 |
| NAB `cpu_asg` (fused) | ~0.815 | ~0.871 |
| NAB `ec2_cpu_ac20cd` (fused) | ~0.570 | ~0.449 |

Additional observations:
- On `cpu_asg`, the **LSTM Autoencoder** reaches **F1 ≈ 0.788 / AUC ≈ 0.871**
- On `cpu_asg`, **Isolation Forest** also performs well (**F1 ≈ 0.706 / AUC ≈ 0.848**)
- On `cpu_asg`, the **LSTM Forecaster** underperforms (**F1 ≈ 0.276**)
- On `ec2_cpu_ac20cd`, the **LSTM Autoencoder** is the strongest individual detector (**F1 ≈ 0.638**)

### Interpretation

- The pipeline generalises well to real CPU-related anomalies when the anomaly manifests as a clear distribution shift.
- Performance degrades on datasets with atypical label distributions, where ranking quality becomes unreliable.
- The **LSTM Autoencoder** is the most consistent model across both synthetic and real-world CPU-centric series.
- The **LSTM Forecaster** performs strongly on synthetic temporal anomalies but is less robust to real-world distribution shifts.

### Conclusion

External validation confirms that:
- the architecture generalises beyond synthetic data
- model performance depends strongly on anomaly structure
- forecasting is strongest on the synthetic benchmark, while reconstruction appears more robust across heterogeneous real-world scenarios

---

## Ablation Insights

- Changing Autoencoder capacity did not materially alter the global conclusion.
- The Forecaster remained the strongest DL model across all tested settings on the synthetic benchmark.
- With larger windows, classical and reconstruction-based approaches degraded more sharply, while the Forecaster remained robust.

---

## Key Takeaways

- Dataset construction strongly impacts anomaly detection performance.
- Window labeling strategy can artificially inflate results.
- Threshold calibration is essential — high ROC-AUC does not guarantee strong thresholded classification.
- A deep learning model does not automatically outperform a well-tuned classical baseline.
- On the synthetic benchmark, **forecasting-based DL outperforms reconstruction-based DL**.
- On external NAB validation, model performance depends strongly on anomaly type.
- Reconstruction and isolation-based approaches appear more robust to real-world distribution shifts.
- Robust multi-seed evaluation is necessary to validate performance claims.

---

## Technical Highlights

- End-to-end anomaly detection pipeline with training/serving separation
- Synthetic time-series benchmark engineering with realistic anomaly injection
- Isolation Forest, LSTM Autoencoder, and LSTM Forecaster implementations
- Validation-based threshold optimisation
- Multi-seed evaluation with metric aggregation
- External validation on NAB CPU-oriented time series
- MLflow experiment tracking
- LangGraph agent for structured LLM-assisted anomaly interpretation
- Performance dashboard: ROC curves, metric comparison, seed-stability plots
- REST API via FastAPI; containerised deployment via Docker
