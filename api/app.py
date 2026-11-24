"""
api/app.py  —  FastAPI REST server for NPTEL Lecture Retrieval
===============================================================
Provides a production-ready HTTP endpoint wrapping the retrieval pipeline.

Endpoints
---------
  GET  /            → service info (version, models, deployed strategies)
  GET  /health      → liveness probe  (process is up)
  GET  /ready       → readiness probe (models + default index loaded)
  POST /search      → run the full retrieval pipeline
  GET  /strategies  → list chunking strategies and which are deployed

Running
-------
    # From the project root:
    uvicorn api.app:app --reload --port 8000

    # Or directly:
    python api/app.py

Environment variables (all optional):
    PROJECT_ROOT             — project root directory (default: repo root)
    EMBEDDING_DEVICE         — "cuda" or "cpu"
    DEFAULT_STRATEGY         — default chunking strategy (default: c3)
    INDEX_STRATEGIES         — strategies to download when DOWNLOAD_INDEXES=1 (default: c1,c2,c3)
    DOWNLOAD_INDEXES         — "1" to pull missing index files from the HF dataset at startup
    INDEX_REPO_ID            — HF dataset holding the indexes (default: SouravNath/vidyarag-indexes)
    MAX_CONCURRENT_SEARCHES  — searches allowed to run at once (default: 2)
    SEARCH_QUEUE_TIMEOUT_SEC — how long a request waits for a free slot before 503 (default: 30)
    APP_VERSION              — build identifier shown on / (the Docker build sets the git SHA)
    OLLAMA_MODEL             — ollama model name (default: llama3.2:3b)

Example request:
    curl -X POST http://localhost:8000/search \\
         -H "Content-Type: application/json" \\
         -d '{"query": "how does BST insertion work", "strategy": "c3", "top_k": 5}'
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
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

# retriever.py resolves data/indexes relative to PROJECT_ROOT, so make sure it
# points at the repo root even when no .env is present (e.g. inside Docker/CI).
os.environ.setdefault("PROJECT_ROOT", str(_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("vidyarag.api")

# ── FastAPI ────────────────────────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as e:
    raise ImportError(
        f"FastAPI not installed: {e}\n"
        "Run: pip install fastapi uvicorn"
    )

# ── Index bootstrap ────────────────────────────────────────────────────────────
# The production Docker image bakes the indexes in at build time, so this is a
# no-op there. On HF Spaces (SPACE_ID set) or when DOWNLOAD_INDEXES=1, missing
# index files are pulled from the HF dataset before `retriever` is imported.
INDEX_REPO_ID    = os.getenv("INDEX_REPO_ID", "SouravNath/vidyarag-indexes")
INDEX_STRATEGIES = [s.strip() for s in os.getenv("INDEX_STRATEGIES", "c1,c2,c3").split(",") if s.strip()]
_IDX_DIR         = Path(os.environ["PROJECT_ROOT"]) / "data" / "indexes"


def _index_files(strategy: str) -> list[str]:
    return [f"faiss_{strategy}.index", f"metadata_{strategy}.json", f"bm25_{strategy}.pkl"]


def _bootstrap_indexes() -> None:
    if not (os.getenv("SPACE_ID") or os.getenv("DOWNLOAD_INDEXES") == "1"):
        return
    missing = [f for s in INDEX_STRATEGIES for f in _index_files(s) if not (_IDX_DIR / f).exists()]
    if not missing:
        return
    log.info("bootstrap: downloading %d index file(s) from %s", len(missing), INDEX_REPO_ID)
    try:
        _IDX_DIR.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import hf_hub_download
    except Exception as e:  # read-only filesystem, missing package, ...
        log.error("bootstrap: cannot prepare index download: %s", e)
        return
    for fname in missing:
        try:
            hf_hub_download(repo_id=INDEX_REPO_ID, repo_type="dataset",
                            filename=fname, local_dir=str(_IDX_DIR))
        except Exception as e:
            log.error("bootstrap: could not download %s: %s", fname, e)


_bootstrap_indexes()

import retriever as _ret  # noqa: E402  (must come after the bootstrap)

# ─────────────────────────────────────────────────────────────────────────────
# Runtime configuration & state
# ─────────────────────────────────────────────────────────────────────────────

APP_VERSION      = os.getenv("APP_VERSION", "dev")
DEFAULT_STRATEGY = os.getenv("DEFAULT_STRATEGY", "c3")
VALID_STRATEGIES = {"c1", "c2", "c3", "c2_w150", "c2_w250", "c3_t025", "c3_t040", "demo"}

# Back-pressure: the pipeline is CPU-bound, so running many searches at once only
# makes every one of them slow (and multiplies memory). Extra requests wait for a
# slot, then get a 503 + Retry-After instead of piling up.
MAX_CONCURRENT_SEARCHES  = int(os.getenv("MAX_CONCURRENT_SEARCHES", "2"))
SEARCH_QUEUE_TIMEOUT_SEC = float(os.getenv("SEARCH_QUEUE_TIMEOUT_SEC", "30"))
_search_slots = threading.BoundedSemaphore(MAX_CONCURRENT_SEARCHES)
_index_lock   = threading.Lock()   # load each strategy's index only once

_state: dict = {"ready": False, "error": None, "startup_sec": None}


def _available_strategies() -> list[str]:
    """Strategies whose index files are actually present on this instance."""
    return sorted(
        s for s in VALID_STRATEGIES
        if (_ret.INDEXES / f"faiss_{s}.index").exists()
        and (_ret.INDEXES / f"metadata_{s}.json").exists()
    )


class IndexMismatchError(RuntimeError):
    """The index was built with a different embedding model than the one loaded."""


def _check_dimensions(strategy: str) -> None:
    # FAISS only fails with a bare `assert d == self.d` deep inside search, so
    # catch a model/index mismatch up front with an actionable message.
    faiss_index = (_ret._index_cache.get(strategy) or {}).get("faiss")
    get_dim = getattr(_ret._get_embed_model(), "get_sentence_embedding_dimension", None)
    if faiss_index is None or get_dim is None:
        return
    model_dim = get_dim()
    if model_dim and faiss_index.d != model_dim:
        _ret._index_cache.pop(strategy, None)
        raise IndexMismatchError(
            f"index '{strategy}' holds {faiss_index.d}-dim vectors but EMBEDDING_MODEL="
            f"{_ret.EMBEDDING_MODEL} produces {model_dim}-dim embeddings"
        )


def _ensure_index_loaded(strategy: str) -> None:
    if strategy in _ret._index_cache:
        return
    with _index_lock:
        if strategy not in _ret._index_cache:
            _ret._load_index(strategy)
            _check_dimensions(strategy)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Warm-up before the server accepts traffic: load the embedding model, the
    reranker and the default strategy's index. Uvicorn only opens the port once
    this finishes, so the first real request doesn't pay the ~30 s load cost.
    """
    t0 = time.time()
    try:
        _ret._get_embed_model()
        _ret._get_rerank_model()
        if DEFAULT_STRATEGY not in _available_strategies():
            raise FileNotFoundError(
                f"no index files for default strategy '{DEFAULT_STRATEGY}' in {_ret.INDEXES}"
            )
        _ensure_index_loaded(DEFAULT_STRATEGY)
        _state["ready"] = True
    except Exception as e:
        _state["error"] = f"{type(e).__name__}: {e}"
        log.exception("warm-up failed — /ready will return 503")
    _state["startup_sec"] = round(time.time() - t0, 1)
    log.info("warm-up finished in %.1fs (ready=%s, strategies=%s)",
             _state["startup_sec"], _state["ready"], _available_strategies())
    yield


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
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_QUIET_PATHS = {"/health", "/ready"}   # probe traffic — keep it out of the logs


@app.middleware("http")
async def access_log(request: Request, call_next):
    """One structured log line per request + an X-Response-Time-ms header."""
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    if request.url.path not in _QUIET_PATHS:
        log.info(json.dumps({
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "ms": round(ms, 1),
        }))
    return response


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=512,
        description="Natural language search query.",
        examples=["how does binary search tree insertion work"],
    )
    strategy: Optional[str] = Field(
        default=None,
        description="Chunking strategy: 'c1' (fixed 30s), 'c2' (utterance), 'c3' (slide-boundary).",
        examples=["c3"],
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
    segment_id: Optional[str] = None
    course_name: Optional[str] = None
    instructor: Optional[str] = None
    lecture_title: Optional[str] = None
    lecture_number: Optional[int] = None
    youtube_url: Optional[str] = None
    youtube_deep_link: Optional[str] = None
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    duration_sec: Optional[float] = None
    transcript: Optional[str] = None
    ocr_text: Optional[str] = None
    content_type: Optional[str] = None
    is_code_segment: Optional[bool] = None
    chunking_strategy: Optional[str] = None
    retrieval_score: float
    query_intent: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    strategy: str
    top_k: int
    latency_sec: float
    n_results: int
    results: List[SearchResultItem]


class InfoResponse(BaseModel):
    status: str
    version: str
    ready: bool
    default_strategy: str
    available_strategies: List[str]
    embedding_model: str
    reranker_model: str


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_model=InfoResponse, tags=["Health"])
def info():
    """Service info: build version, models and which strategies this instance serves."""
    return InfoResponse(
        status="ok",
        version=APP_VERSION,
        ready=_state["ready"],
        default_strategy=DEFAULT_STRATEGY,
        available_strategies=_available_strategies(),
        embedding_model=_ret.EMBEDDING_MODEL,
        reranker_model=_ret.RERANKER_MODEL,
    )


@app.get("/health", tags=["Health"])
def health():
    """Liveness probe — the process is up and serving HTTP."""
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
def ready():
    """Readiness probe — 200 only once models and the default index are loaded."""
    if _state["ready"]:
        return {"status": "ready", "startup_sec": _state["startup_sec"], "version": APP_VERSION}
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "error": _state["error"]},
    )


@app.get("/strategies", tags=["Info"])
def list_strategies():
    """Lists all chunking strategies, and which ones are deployed on this instance."""
    return {
        "strategies": {
            "c1": "Fixed 30-second window chunks. Simple baseline.",
            "c2": "Word-count boundary chunks (≈200 words). Respects sentence structure.",
            "c3": "Slide-boundary chunks via OCR Jaccard similarity. Best performance.",
            "c2_w150": "C2 variant — 150-word target (chunk sensitivity experiment).",
            "c2_w250": "C2 variant — 250-word target (chunk sensitivity experiment).",
            "c3_t025": "C3 variant — threshold=0.25, more sensitive to slide changes.",
            "c3_t040": "C3 variant — threshold=0.40, less sensitive to slide changes.",
            "demo": "300-segment sample index (needs EMBEDDING_MODEL=all-MiniLM-L6-v2).",
        },
        "available": _available_strategies(),
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
    available = _available_strategies()
    if strategy not in available:
        raise HTTPException(
            status_code=503,
            detail=f"Strategy '{strategy}' is not deployed on this instance. Available: {available}",
        )

    if not _search_slots.acquire(timeout=SEARCH_QUEUE_TIMEOUT_SEC):
        raise HTTPException(
            status_code=503,
            detail="Server busy — too many searches in progress. Please retry shortly.",
            headers={"Retry-After": "5"},
        )
    try:
        t0 = time.time()
        _ensure_index_loaded(strategy)
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
        latency = round(time.time() - t0, 3)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Index not found: {e}")
    except IndexMismatchError as e:
        raise HTTPException(status_code=503, detail=f"Strategy '{strategy}' can't be served: {e}")
    except Exception as e:
        log.exception("search failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _search_slots.release()

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
    import uvicorn  # only needed to self-host; importing the app doesn't require it

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )
