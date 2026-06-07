# ── Dockerfile for FastAPI endpoint (HuggingFace Spaces / Docker) ─────────────
# Build:  docker build -t nptel-retrieval-api .
# Run:    docker run -p 8000:8000 nptel-retrieval-api
# HF:     Push to a Docker-based HuggingFace Space

FROM python:3.10-slim

WORKDIR /app

# Install system deps for Tesseract OCR and OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Set project root so retriever can find indexes
ENV PROJECT_ROOT=/app
ENV EMBEDDING_DEVICE=cpu

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
