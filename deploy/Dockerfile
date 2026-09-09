# ── Dockerfile for FastAPI endpoint (HuggingFace Spaces / Docker) ─────────────
# Build:  docker build -f deploy/Dockerfile -t nptel-retrieval-api .   (from repo root)
# Run:    docker run -p 8000:7860 nptel-retrieval-api   (visit localhost:8000)
# HF:     Push to a Docker-based HuggingFace Space (auto-uses port 7860)

FROM python:3.10-slim

WORKDIR /app

# Copy requirements first (layer caching) — lean API-only deps, no
# Whisper/yt-dlp/OpenCV/Tesseract (those are offline-pipeline only,
# see requirements.txt for the full data-collection stack)
COPY deploy/requirements/api.txt .
RUN pip install --no-cache-dir -r api.txt

# Copy only what the API needs at runtime
COPY api/ api/
COPY src/ src/
COPY configs/ configs/

# Expose FastAPI port (7860 = HF Spaces Docker SDK default)
EXPOSE 7860

# Set project root so retriever can find indexes
ENV PROJECT_ROOT=/app
ENV EMBEDDING_DEVICE=cpu

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
