FROM python:3.11-slim

# tensorflow-cpu requires libgomp1 for OpenMP threading
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the dependency manifest first so pip-install is cached separately
COPY pyproject.toml ./
# Install a minimal stub so that pip can resolve the package metadata before
# the source tree is present, then install the real package below.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[dev]" || true

COPY anomaliz/ anomaliz/
RUN pip install --no-cache-dir -e .

EXPOSE 8000

# ANOMALIZ_ARTIFACT_DIR must be set (via env or docker-compose) to a directory
# that contains a trained bundle before the API can serve requests.
CMD ["uvicorn", "anomaliz.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
