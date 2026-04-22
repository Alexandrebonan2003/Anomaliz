FROM python:3.11-slim

# tensorflow-cpu requires libgomp1 for OpenMP threading
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the dependency manifest first so pip-install is cached separately
COPY pyproject.toml ./
# Stub lets pip resolve metadata and install all deps before the real source
# is copied, preserving layer caching without swallowing errors.
RUN pip install --no-cache-dir --upgrade pip \
    && mkdir -p anomaliz \
    && touch anomaliz/__init__.py \
    && pip install --no-cache-dir -e ".[dev]"

COPY anomaliz/ anomaliz/

EXPOSE 8000

# ANOMALIZ_ARTIFACT_DIR must be set (via env or docker-compose) to a directory
# that contains a trained bundle before the API can serve requests.
CMD ["uvicorn", "anomaliz.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
