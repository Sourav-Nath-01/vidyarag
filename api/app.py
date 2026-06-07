"""
api/app.py  —  FastAPI REST server for NPTEL Lecture Retrieval
===============================================================
Provides a production-ready HTTP endpoint wrapping the retrieval pipeline.

Endpoints
---------
  GET  /            → health check + system info
  POST /search      → run the full retrieval pipeline
  GET  /strategies  → list available chunking strategies

Running
-------
    # From the project root:
    uvicorn api.app:app --reload --port 8000

    # Or directly:
    python api/app.py

Environment variables (from .env):
    PROJECT_ROOT     — project root directory
    EMBEDDING_DEVICE — "cuda" or "cpu"
    OLLAMA_MODEL     — ollama model name (default: llama3.2:3b)
    DEFAULT_STRATEGY — default chunking strategy (default: c2)

Example request:
    curl -X POST http://localhost:8000/search \\
         -H "Content-Type: application/json" \\
         -d '{"query": "how does BST insertion work", "strategy": "c3", "top_k": 5}'
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

# ── path setup ─────────────────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
_root = _here.parent
for _candidate in [_root / "src" / "retrieval", _root / "src"]:
    if (_candidate / "retriever.py").exists():
        sys.path.insert(0, str(_candidate))
        break

# ── load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    for _c in [_root, _root.parent]:
        if (_c / ".env").exists():
            load_dotenv(_c / ".env")
            break
except ImportError:
    pass

# ── FastAPI ────────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError as e:
    raise ImportError(
        f"FastAPI not installed: {e}\n"
        "Run: pip install fastapi uvicorn"
    )

import retriever as _ret

# ─────────────────────────────────────────────────────────────────────────────
# App initialisation
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VidyaRAG — NPTEL Lecture Retrieval API",
    description=(
        "Multimodal hybrid retrieval over NPTEL lecture transcripts and OCR slide text. "
        "Pipeline: BGE-large dense (FAISS) + BM25 sparse, fused via Reciprocal Rank Fusion, "
        "reranked with cross-encoder/ms-marco-MiniLM. Achieves MRR 0.826 / Recall@10 0.964."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Warm up embedding model on startup (load once, serve many)
@app.on_event("startup")
def warmup():
    try:
        _ret._get_embed_model()
    except Exception as e:
        print(f"[WARNING] Could not warm up embedding model: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=512,
        description="Natural language search query.",
        example="how does binary search tree insertion work",
    )
    strategy: Optional[str] = Field(
        default=None,
        description="Chunking strategy: 'c1' (fixed 30s), 'c2' (utterance), 'c3' (slide-boundary).",
        example="c3",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of unique lecture results to return.",
    )
    use_rerank: bool = Field(
        default=True,
        description="Apply cross-encoder reranking (ms-marco-MiniLM). Slower but more accurate.",
    )
    use_bm25: bool = Field(
        default=True,
        description="Enable BM25 sparse retrieval fused with dense via RRF.",
    )
    use_ocr: bool = Field(
        default=True,
        description="Include OCR slide text in the reranker passage.",
    )
    use_llm: bool = Field(
        default=False,
        description="Use Ollama LLM for query intent classification and expansion. Requires `ollama serve`.",
    )
    course_filter: Optional[List[str]] = Field(
        default=None,
        description="Optional list of course names to restrict results to.",
    )


class SearchResultItem(BaseModel):
    rank: int
    segment_id: Optional[str]
    course_name: Optional[str]
    instructor: Optional[str]
    lecture_title: Optional[str]
    lecture_number: Optional[int]
    youtube_url: Optional[str]
    youtube_deep_link: Optional[str]
    start_sec: Optional[float]
    end_sec: Optional[float]
    duration_sec: Optional[float]
    transcript: Optional[str]
    ocr_text: Optional[str]
    content_type: Optional[str]
    is_code_segment: Optional[bool]
    chunking_strategy: Optional[str]
    retrieval_score: float
    query_intent: Optional[str]


class SearchResponse(BaseModel):
    query: str
    strategy: str
    top_k: int
    latency_sec: float
    n_results: int
    results: List[SearchResultItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    available_strategies: List[str]
    embedding_model: str
    reranker_model: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

VALID_STRATEGIES = {"c1", "c2", "c3", "c2_w150", "c2_w250", "c3_t025", "c3_t040"}
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "c2")


@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Returns API status and model configuration."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        available_strategies=sorted(VALID_STRATEGIES),
        embedding_model=_ret.EMBEDDING_MODEL,
        reranker_model=_ret.RERANKER_MODEL,
    )


@app.get("/strategies", tags=["Info"])
def list_strategies():
    """Lists all available chunking strategies and their descriptions."""
    return {
        "strategies": {
            "c1": "Fixed 30-second window chunks. Simple baseline.",
            "c2": "Word-count boundary chunks (≈200 words). Respects sentence structure.",
            "c3": "Slide-boundary chunks via OCR Jaccard similarity. Best performance.",
            "c2_w150": "C2 variant — 150-word target (chunk sensitivity experiment).",
            "c2_w250": "C2 variant — 250-word target (chunk sensitivity experiment).",
            "c3_t025": "C3 variant — threshold=0.25, more sensitive to slide changes.",
            "c3_t040": "C3 variant — threshold=0.40, less sensitive to slide changes.",
        },
        "recommended": "c3",
        "best_metrics": {
            "MRR": 0.8259,
            "Recall@5": 0.9524,
            "Recall@10": 0.9643,
        },
    }


@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
def search(request: SearchRequest):
    """
    Run the full NPTEL lecture retrieval pipeline.

    Pipeline: query → [LLM expand] → BGE-large embed → FAISS → [BM25] → RRF → boost → rerank → dedup → results
    """
    strategy = request.strategy or DEFAULT_STRATEGY

    if strategy not in VALID_STRATEGIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown strategy '{strategy}'. Valid: {sorted(VALID_STRATEGIES)}",
        )

    t0 = time.time()
    try:
        results = _ret.search(
            query      = request.query,
            strategy   = strategy,
            top_k      = request.top_k,
            use_llm    = request.use_llm,
            use_rerank = request.use_rerank,
            use_ocr    = request.use_ocr,
            use_bm25   = request.use_bm25,
            verbose    = False,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Index not found: {e}. Run embedder.py and bm25_builder.py first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency = round(time.time() - t0, 3)

    # Apply optional course filter
    if request.course_filter:
        results = [
            r for r in results
            if r.get("course_name") in request.course_filter
        ]

    return SearchResponse(
        query       = request.query,
        strategy    = strategy,
        top_k       = request.top_k,
        latency_sec = latency,
        n_results   = len(results),
        results     = [SearchResultItem(**r) for r in results],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (direct run)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
