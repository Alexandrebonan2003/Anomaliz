PYTHON  := .venv/bin/python
PIP     := .venv/bin/pip
PYTEST  := .venv/bin/pytest
UVICORN := .venv/bin/uvicorn

BUNDLE  ?= artifacts/dev
REPORTS ?= reports

.DEFAULT_GOAL := help

# ── setup ─────────────────────────────────────────────────────────────────────

.PHONY: install
install:  ## Create .venv and install all dependencies
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

# ── training ──────────────────────────────────────────────────────────────────

.PHONY: train
train:  ## Train all detectors and write bundle to $(BUNDLE)
	$(PYTHON) -m anomaliz.training.cli --out $(BUNDLE)

.PHONY: train-sweep
train-sweep:  ## Train + run capacity/window ablations
	$(PYTHON) -m anomaliz.training.cli --out $(BUNDLE) --sweep

.PHONY: train-mlflow
train-mlflow:  ## Train and log the run to MLflow (SQLite store)
	$(PYTHON) -m anomaliz.training.cli --out $(BUNDLE) \
	  --logger mlflow \
	  --tracking-uri sqlite:///mlruns.db

# ── serving ───────────────────────────────────────────────────────────────────

.PHONY: serve
serve:  ## Start the FastAPI server (requires a trained bundle at $(BUNDLE))
	ANOMALIZ_ARTIFACT_DIR=$(BUNDLE) \
	$(UVICORN) anomaliz.api.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: serve-with-agent
serve-with-agent:  ## Start the API with the Ollama-backed explainability agent
	ANOMALIZ_ARTIFACT_DIR=$(BUNDLE) \
	ANOMALIZ__AGENT__BACKEND=ollama \
	$(UVICORN) anomaliz.api.main:app --host 0.0.0.0 --port 8000 --reload

# ── evaluation & inspection ───────────────────────────────────────────────────

.PHONY: dashboard
dashboard:  ## Generate visualisation PNGs from bundle metrics into $(REPORTS)/
	$(PYTHON) -m anomaliz.visualization.dashboard --bundle $(BUNDLE) --out $(REPORTS)

.PHONY: mlflow-ui
mlflow-ui:  ## Open the MLflow experiment UI (SQLite backend)
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

# ── docker ────────────────────────────────────────────────────────────────────

.PHONY: docker-up
docker-up:  ## Start API + MLflow UI via docker-compose
	docker compose up --build

.PHONY: docker-up-llm
docker-up-llm:  ## Start API + MLflow UI + Ollama
	docker compose --profile llm up --build

.PHONY: docker-down
docker-down:  ## Stop and remove all containers
	docker compose down

# ── evaluation ────────────────────────────────────────────────────────────────

.PHONY: eval-nab
eval-nab:  ## Evaluate trained models on NAB real-world data (downloads ~100 KB)
	$(PYTHON) -m anomaliz.data.nab --bundle $(BUNDLE) --cache-dir .nab_cache

# ── tests ─────────────────────────────────────────────────────────────────────

.PHONY: test
test:  ## Run the full pytest suite
	$(PYTEST)

.PHONY: test-fast
test-fast:  ## Run tests excluding the slow training smoke tests
	$(PYTEST) --ignore=tests/test_training_smoke.py -q

# ── help ──────────────────────────────────────────────────────────────────────

.PHONY: help
help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
