"""
tests/test_api.py  —  HTTP-level tests for the FastAPI service (api/app.py)
==========================================================================
The real `retriever` module is swapped for a tiny in-memory fake, so these
tests need only fastapi + httpx — no models, no indexes, runs in ~1 s in CI.

What's covered: liveness/readiness probes, startup warm-up, input validation,
strategy availability (422 vs 503), course filtering, response-time header,
and the concurrency limit (503 + Retry-After when all search slots are busy).
"""

from __future__ import annotations

import importlib
import sys
import tempfile
import types
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent

FAKE_RESULTS = [
    {"rank": 1, "course_name": "DSA", "lecture_title": "Binary Search Trees",
     "retrieval_score": 0.91, "start_sec": 120.0, "chunking_strategy": "c3"},
    {"rank": 2, "course_name": "DAA", "lecture_title": "Balanced BSTs",
     "retrieval_score": 0.77, "start_sec": 30.0, "chunking_strategy": "c3"},
]


def _make_fake_retriever(index_dir: Path) -> types.ModuleType:
    fake = types.ModuleType("retriever")
    fake.INDEXES = index_dir
    fake.EMBEDDING_MODEL = "fake-embedder"
    fake.RERANKER_MODEL = "fake-reranker"
    fake._index_cache = {}
    fake.calls = []
    fake._get_embed_model = lambda: object()
    fake._get_rerank_model = lambda: object()

    def _load_index(strategy):
        fake._index_cache[strategy] = {}
        return {}

    def search(query, strategy, top_k, **kwargs):
        fake.calls.append({"query": query, "strategy": strategy, "top_k": top_k, **kwargs})
        return [dict(r) for r in FAKE_RESULTS[:top_k]]

    fake._load_index = _load_index
    fake.search = search
    return fake


@pytest.fixture(scope="module")
def api():
    """Import api.app against the fake retriever; only the c3 index 'exists'."""
    index_dir = Path(tempfile.mkdtemp())
    for name in ("faiss_c3.index", "metadata_c3.json"):
        (index_dir / name).write_text("stub")

    mp = pytest.MonkeyPatch()
    mp.setenv("DEFAULT_STRATEGY", "c3")
    mp.setenv("SEARCH_QUEUE_TIMEOUT_SEC", "0.1")
    mp.setenv("APP_VERSION", "test-sha")
    mp.delenv("SPACE_ID", raising=False)
    mp.delenv("DOWNLOAD_INDEXES", raising=False)
    mp.syspath_prepend(str(_ROOT))
    mp.setitem(sys.modules, "retriever", _make_fake_retriever(index_dir))
    sys.modules.pop("api.app", None)
    try:
        yield importlib.import_module("api.app")
    finally:
        # Always restore the real `retriever` — even if the import above
        # failed — so other test modules never see the fake.
        sys.modules.pop("api.app", None)
        mp.undo()


@pytest.fixture(scope="module")
def client(api):
    with TestClient(api.app) as c:   # `with` runs the lifespan (warm-up)
        yield c


# ── probes ─────────────────────────────────────────────────────────────────────

def test_health_is_always_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_after_warmup_loads_default_index(client, api):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert "c3" in api._ret._index_cache        # warm-up pre-loaded it


def test_ready_returns_503_when_warmup_failed(client, api, monkeypatch):
    monkeypatch.setitem(api._state, "ready", False)
    monkeypatch.setitem(api._state, "error", "FileNotFoundError: boom")
    r = client.get("/ready")
    assert r.status_code == 503
    assert "boom" in r.json()["error"]


def test_info_reports_version_and_deployed_strategies(client):
    body = client.get("/").json()
    assert body["version"] == "test-sha"
    assert body["ready"] is True
    assert body["default_strategy"] == "c3"
    assert body["available_strategies"] == ["c3"]


# ── search ─────────────────────────────────────────────────────────────────────

def test_search_happy_path(client, api):
    r = client.post("/search", json={"query": "bst insertion", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["strategy"] == "c3"                 # falls back to DEFAULT_STRATEGY
    assert body["n_results"] == 2
    assert body["results"][0]["lecture_title"] == "Binary Search Trees"
    assert float(r.headers["X-Response-Time-ms"]) >= 0
    assert api._ret.calls[-1]["top_k"] == 2


def test_search_course_filter(client):
    r = client.post("/search", json={"query": "bst insertion", "top_k": 2, "course_filter": ["DAA"]})
    assert r.status_code == 200
    assert [x["course_name"] for x in r.json()["results"]] == ["DAA"]


def test_unknown_strategy_is_422(client):
    r = client.post("/search", json={"query": "bst insertion", "strategy": "c9"})
    assert r.status_code == 422


def test_known_but_undeployed_strategy_is_503(client):
    r = client.post("/search", json={"query": "bst insertion", "strategy": "c1"})
    assert r.status_code == 503
    assert "Available" in r.json()["detail"]


@pytest.mark.parametrize("payload", [
    {"query": "x"},                         # too short
    {"query": "bst insertion", "top_k": 0}, # out of range
    {"top_k": 3},                           # missing query
])
def test_invalid_input_is_422(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_model_index_dimension_mismatch_is_a_clear_503(client, api, monkeypatch):
    stubs = [api._ret.INDEXES / n for n in ("faiss_demo.index", "metadata_demo.json")]
    for p in stubs:
        p.write_text("stub")
    model = types.SimpleNamespace(get_sentence_embedding_dimension=lambda: 384)
    monkeypatch.setattr(api._ret, "_get_embed_model", lambda: model)

    def load(strategy):   # an index built with a 1024-dim model (BGE-large)
        api._ret._index_cache[strategy] = {"faiss": types.SimpleNamespace(d=1024)}

    monkeypatch.setattr(api._ret, "_load_index", load)
    try:
        r = client.post("/search", json={"query": "bst insertion", "strategy": "demo"})
        assert r.status_code == 503
        assert "1024-dim" in r.json()["detail"] and "384-dim" in r.json()["detail"]
        assert "demo" not in api._ret._index_cache     # broken index isn't cached
    finally:
        for p in stubs:
            p.unlink()


def test_busy_server_returns_503_with_retry_after(client, api):
    held = 0
    while api._search_slots.acquire(blocking=False):   # occupy every slot
        held += 1
    try:
        r = client.post("/search", json={"query": "bst insertion"})
        assert r.status_code == 503
        assert r.headers.get("Retry-After") == "5"
    finally:
        for _ in range(held):
            api._search_slots.release()
    assert client.post("/search", json={"query": "bst insertion"}).status_code == 200
