"""
app_zerogpu.py  —  VidyaRAG API on Hugging Face ZeroGPU
=========================================================
Entry point for a *second* live deployment: a GPU-accelerated REST API +
minimal search UI, running on HF's free ZeroGPU hardware (on-demand A10G,
released after each call). This is separate from the always-on Streamlit
demo (app.py, CPU-only) — see README for why both exist.

Quota note (HF free tier, as of 2026): 3.5 GPU-minutes/day per account.
Each search is budgeted at 25s, so this supports roughly 8 searches/day
before quota resets 24h after first use. That's a deliberate tradeoff:
genuinely free GPU inference, at the cost of a small daily request cap.
The Streamlit Space has no such cap and is the primary demo link.

Running locally (CPU fallback, no @spaces.GPU allocation needed):
    python app_zerogpu.py
"""

import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
for _candidate in [_here / "src" / "retrieval", _here / "src"]:
    if (_candidate / "retriever.py").exists():
        sys.path.insert(0, str(_candidate))
        break

os.environ.setdefault("PROJECT_ROOT", str(_here))
os.environ.setdefault("EMBEDDING_DEVICE", "cuda" if os.environ.get("SPACE_ID") else "cpu")
os.environ.setdefault("DEFAULT_STRATEGY", "c3")

try:
    from dotenv import load_dotenv
    if (_here / ".env").exists():
        load_dotenv(_here / ".env", override=False)
except ImportError:
    pass


# ── HF Spaces: download full BGE-large indexes from the dataset repo ──────────
def _bootstrap_indexes():
    if not os.environ.get("SPACE_ID"):
        return

    idx_dir = _here / "data" / "indexes"
    idx_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "faiss_c1.index", "faiss_c2.index", "faiss_c3.index",
        "bm25_c1.pkl",    "bm25_c2.pkl",    "bm25_c3.pkl",
        "metadata_c1.json", "metadata_c2.json", "metadata_c3.json",
    ]
    missing = [f for f in required if not (idx_dir / f).exists()]
    if not missing:
        return

    print(f"[bootstrap] Downloading {len(missing)} missing index files...")
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        import shutil

        try:
            snapshot_download(
                repo_id="SouravNath/vidyarag-indexes",
                repo_type="dataset",
                local_dir=str(idx_dir),
                ignore_patterns=["*.gitattributes", ".gitattributes"],
            )
        except Exception as snap_err:
            print(f"[bootstrap] Bulk download incomplete ({snap_err}). Trying per-file fallback...")

        for fname in [f for f in required if not (idx_dir / f).exists()]:
            try:
                tmp = hf_hub_download(
                    repo_id="SouravNath/vidyarag-indexes",
                    repo_type="dataset",
                    filename=fname,
                )
                shutil.copy2(tmp, idx_dir / fname)
            except Exception as file_err:
                print(f"[bootstrap] Could not download {fname}: {file_err}")
    except Exception as e:
        print(f"[bootstrap] Index download failed: {e}")


_bootstrap_indexes()

_idx_dir = _here / "data" / "indexes"
if all((_idx_dir / f).exists() for f in ["faiss_c3.index", "bm25_c3.pkl", "metadata_c3.json"]):
    os.environ["EMBEDDING_MODEL"] = "BAAI/bge-large-en-v1.5"

import retriever as _ret  # noqa: E402

import gradio as gr  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from typing import List, Optional  # noqa: E402

try:
    import spaces
    _HAS_ZEROGPU = True
except ImportError:
    _HAS_ZEROGPU = False

    class _NoOpSpaces:
        @staticmethod
        def GPU(*a, **kw):
            def _decorator(fn):
                return fn
            return _decorator if not (a and callable(a[0])) else a[0]

    spaces = _NoOpSpaces()


# ── Warm up models OUTSIDE the @spaces.GPU function (required by ZeroGPU) ─────
if os.environ.get("SPACE_ID"):
    print("[warmup] Loading embedding + reranker models...")
    try:
        _ret._get_embed_model()
        _ret._get_rerank_model()
    except Exception as e:
        print(f"[warmup] Could not warm up models: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GPU-allocated search — this is the ONLY function that requests the GPU
# ─────────────────────────────────────────────────────────────────────────────

@spaces.GPU(duration=25)
def _gpu_search(query, strategy, top_k, use_rerank, use_bm25):
    return _ret.search(
        query=query,
        strategy=strategy,
        top_k=top_k,
        use_rerank=use_rerank,
        use_bm25=use_bm25,
        verbose=False,
    )


VALID_STRATEGIES = {"c1", "c2", "c3"}


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI — the clickable demo
# ─────────────────────────────────────────────────────────────────────────────

def _format_result(r: dict) -> str:
    ts = f"{int(r.get('start_sec', 0)) // 60}:{int(r.get('start_sec', 0)) % 60:02d}"
    link = r.get("youtube_deep_link", "")
    snippet = (r.get("transcript", "") or "").strip().replace("\n", " ")[:200]
    return (
        f"**#{r['rank']} — {r.get('lecture_title', 'Unknown')}** "
        f"({r.get('course_name', '')})  \n"
        f"⏱ {ts}  ·  score: {r.get('retrieval_score', 0):.4f}  \n"
        f"_{snippet}..._  \n"
        f"[▶ Play from here]({link})\n\n---"
    )


def gradio_search(query, strategy, top_k):
    if not query or not query.strip():
        return "Enter a query to search."
    try:
        results = _gpu_search(query.strip(), strategy, int(top_k), True, True)
    except Exception as e:
        return f"Error: {e}"
    if not results:
        return "No results found."
    return "\n".join(_format_result(r) for r in results)


with gr.Blocks(title="VidyaRAG API — ZeroGPU") as demo:
    gr.Markdown(
        "# 🎓 VidyaRAG — GPU-accelerated Search API\n"
        "Multimodal hybrid retrieval over NPTEL lectures — BGE-large + BM25 + RRF + cross-encoder, "
        "running on free HF ZeroGPU hardware.\n\n"
        "⚠️ **Free-tier quota**: ~8 searches/day (HF caps free GPU time at 3.5 min/day). "
        "For unlimited access, use the [always-on CPU demo](https://huggingface.co/spaces/SouravNath/vidyarag) instead.\n\n"
        "REST API: `POST /api/search` · [API docs](./docs)"
    )
    with gr.Row():
        query_box = gr.Textbox(label="Search query", placeholder="how does binary search tree insertion work", scale=4)
        strategy_box = gr.Dropdown(choices=["c3", "c2", "c1"], value="c3", label="Strategy", scale=1)
        topk_box = gr.Slider(1, 10, value=5, step=1, label="Results", scale=1)
    search_btn = gr.Button("Search", variant="primary")
    output = gr.Markdown()

    search_btn.click(gradio_search, [query_box, strategy_box, topk_box], output)
    query_box.submit(gradio_search, [query_box, strategy_box, topk_box], output)


# ─────────────────────────────────────────────────────────────────────────────
# REST API — mounted on the same FastAPI app Gradio serves
# ─────────────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=512)
    strategy: str = Field(default="c3")
    top_k: int = Field(default=5, ge=1, le=10)
    use_rerank: bool = Field(default=True)
    use_bm25: bool = Field(default=True)


class SearchResultItem(BaseModel):
    rank: int
    course_name: Optional[str] = None
    lecture_title: Optional[str] = None
    youtube_deep_link: Optional[str] = None
    start_sec: Optional[float] = None
    transcript: Optional[str] = None
    retrieval_score: float
    query_intent: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    strategy: str
    n_results: int
    results: List[SearchResultItem]


fastapi_app = demo.app


@fastapi_app.get("/api/health")
def api_health():
    return {
        "status": "ok",
        "hardware": "ZeroGPU (A10G, on-demand)" if _HAS_ZEROGPU else "cpu (local)",
        "quota_note": "Free tier: ~3.5 GPU-min/day (~8 searches). Resets 24h after first use.",
    }


@fastapi_app.post("/api/search", response_model=SearchResponse)
def api_search(request: SearchRequest):
    if request.strategy not in VALID_STRATEGIES:
        raise HTTPException(422, f"Unknown strategy '{request.strategy}'. Valid: {sorted(VALID_STRATEGIES)}")
    try:
        results = _gpu_search(request.query, request.strategy, request.top_k, request.use_rerank, request.use_bm25)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Index not available: {e}")
    except Exception as e:
        raise HTTPException(500, str(e))
    return SearchResponse(
        query=request.query,
        strategy=request.strategy,
        n_results=len(results),
        results=[SearchResultItem(**r) for r in results],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
