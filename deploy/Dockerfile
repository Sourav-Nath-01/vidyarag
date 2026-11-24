# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# VidyaRAG FastAPI service — production image
#
#   Build (from repo root):  docker build -f deploy/Dockerfile -t vidyarag-api .
#   Run:                     docker run --rm -p 8000:7860 --memory 4g vidyarag-api
#   Check:                   python deploy/smoke_test.py http://localhost:8000
#
# Design notes
#   • CPU-only PyTorch wheel → ~3 GB smaller than the default CUDA build.
#   • Model weights + C3 indexes are baked in at build time: the image is
#     immutable and self-contained (runs with --network none), and cold starts
#     never depend on a download from the Hugging Face Hub.
#   • Layers are ordered from least to most frequently changed — OS deps,
#     Python deps, ~1.6 GB of artifacts, then app code — so a code change only
#     rebuilds the last few-KB layers.
#   • Runs as non-root uid 1000 (also what Hugging Face Docker Spaces expect).
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/Sourav-Nath-01/vidyarag" \
      org.opencontainers.image.description="VidyaRAG — hybrid lecture retrieval API (FastAPI)" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# OpenMP runtime used by faiss-cpu and torch
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app \
 && chown app:app /app
WORKDIR /app

# ── 1) Python dependencies ────────────────────────────────────────────────────
COPY deploy/requirements/api.txt /tmp/api.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu "$(grep -E '^torch==' /tmp/api.txt)" \
 && pip install -r /tmp/api.txt

USER app
ENV HOME=/home/app \
    HF_HOME=/app/.cache/huggingface \
    PROJECT_ROOT=/app \
    EMBEDDING_DEVICE=cpu \
    EMBEDDING_MODEL=BAAI/bge-large-en-v1.5 \
    RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2 \
    DEFAULT_STRATEGY=c3

# ── 2) Model weights + search indexes (~1.6 GB) ───────────────────────────────
# Only C3 (the best strategy) by default — it keeps peak memory well inside a
# 4 GiB container. Build with --build-arg INDEX_STRATEGIES=c1,c2,c3 for all.
ARG INDEX_STRATEGIES=c3
ARG INDEX_REPO_ID=SouravNath/vidyarag-indexes
COPY --chown=app:app scripts/fetch_artifacts.py scripts/fetch_artifacts.py
RUN python scripts/fetch_artifacts.py --strategies "${INDEX_STRATEGIES}" --index-repo "${INDEX_REPO_ID}"

# ── 3) Application code (small, changes most often → last) ───────────────────
COPY --chown=app:app src/retrieval/ src/retrieval/
COPY --chown=app:app api/ api/

# Everything the app needs is in the image now — forbid runtime Hub calls.
# OMP_NUM_THREADS matches the 2 vCPUs the container gets on Azure, so torch
# doesn't oversubscribe the host's cores.
ARG GIT_SHA=dev
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=2 \
    PORT=7860 \
    APP_VERSION=${GIT_SHA}

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ['PORT'], timeout=4)"

# exec → uvicorn becomes PID 1 and receives SIGTERM for a graceful shutdown
CMD ["sh", "-c", "exec uvicorn api.app:app --host 0.0.0.0 --port ${PORT}"]
