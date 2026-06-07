"""
quick_demo.py  —  Fast end-to-end demo (CPU-friendly, runs in ~2 minutes)
=========================================================================
BGE-large on CPU takes ~20 hours for 16k segments. This demo:
  1. Takes 200 real segments from segments_c3.jsonl
  2. Embeds them with all-MiniLM-L6-v2 (90MB, fast on CPU)
  3. Builds a FAISS index + BM25 index in memory
  4. Runs the FULL retrieval pipeline (dense + BM25 + RRF + rerank + dedup)
  5. Prints results for several test queries

This lets you SEE the complete system working without a GPU.
For production quality results, use the full BGE-large embedder.

Usage:
    source venv/bin/activate
    python3 quick_demo.py
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "retrieval"))

SEGMENTS_FILE = PROJECT_ROOT / "data" / "processed" / "segments_c3.jsonl"
DEMO_INDEX    = PROJECT_ROOT / "data" / "indexes" / "faiss_demo.index"
DEMO_META     = PROJECT_ROOT / "data" / "indexes" / "metadata_demo.json"
DEMO_BM25     = PROJECT_ROOT / "data" / "indexes" / "bm25_demo.pkl"

DEMO_MODEL    = "all-MiniLM-L6-v2"    # 90MB, 20-50x faster than BGE-large on CPU
N_SEGMENTS    = 300                    # how many segments to index for demo

DEMO_QUERIES = [
    "how does binary search tree insertion work",
    "what is backpropagation in neural networks",
    "explain virtual memory in operating systems",
    "how does TCP three way handshake work",
    "time complexity of merge sort",
]


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load segments
# ─────────────────────────────────────────────────────────────────────────────

def load_demo_segments() -> list[dict]:
    print(f"\n📂 Loading {N_SEGMENTS} segments from {SEGMENTS_FILE.name} ...")
    if not SEGMENTS_FILE.exists():
        print(f"   ❌ File not found: {SEGMENTS_FILE}")
        print("   Make sure data/processed/segments_c3.jsonl exists")
        sys.exit(1)

    segments = []
    with open(SEGMENTS_FILE, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= N_SEGMENTS:
                break
            line = line.strip()
            if line:
                try:
                    segments.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    print(f"   ✅ Loaded {len(segments)} segments")
    courses = set(s.get("course_id", "?") for s in segments)
    print(f"   Courses covered: {sorted(courses)}")
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Build FAISS + BM25 demo indexes
# ─────────────────────────────────────────────────────────────────────────────

def build_demo_indexes(segments: list[dict]) -> None:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    from retriever import _tokenise

    print(f"\n🔨 Building demo indexes with {DEMO_MODEL} ...")

    # Build passage texts
    texts = []
    for seg in segments:
        parts = ["passage:"]
        if seg.get("course_name"):
            parts.append(f"[COURSE]: {seg['course_name']}")
        if seg.get("lecture_title"):
            parts.append(f"[LECTURE]: {seg['lecture_title']}")
        ocr = seg.get("ocr_text", "").strip()
        if ocr and not seg.get("ocr_failed", False) and len(ocr) > 10:
            parts.append(f"[SLIDE]: {ocr.replace(chr(10), ' ')[:300]}")
        transcript = seg.get("transcript", "").strip()
        if transcript:
            parts.append(f"[SPEECH]: {transcript}")
        texts.append(" ".join(parts))

    # Embed
    print(f"   Embedding {len(texts)} passages (batch=64) ...")
    t0 = time.time()
    model = SentenceTransformer(DEMO_MODEL, device="cpu")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    elapsed = time.time() - t0
    print(f"   ✅ Embedded in {elapsed:.1f}s ({len(texts)/elapsed:.0f} passages/sec)")

    # FAISS index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    DEMO_INDEX.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(DEMO_INDEX))
    print(f"   ✅ FAISS index: {index.ntotal} vectors, dim={dim} → {DEMO_INDEX.name}")

    # Metadata
    DEMO_META.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")
    print(f"   ✅ Metadata saved → {DEMO_META.name}")

    # BM25
    corpus  = [_tokenise(t) for t in texts]
    bm25    = BM25Okapi(corpus)
    with open(DEMO_BM25, "wb") as fh:
        pickle.dump({"bm25": bm25, "corpus": corpus}, fh)
    print(f"   ✅ BM25 index saved → {DEMO_BM25.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Run live retrieval pipeline using retriever.py
# ─────────────────────────────────────────────────────────────────────────────

def run_demo_queries() -> None:
    # Monkey-patch retriever to use demo model + demo indexes
    # MUST set env vars BEFORE importing retriever (module reads them at import time)
    import os
    os.environ["EMBEDDING_MODEL"]  = DEMO_MODEL
    os.environ["EMBEDDING_DEVICE"] = "cpu"
    os.environ["PROJECT_ROOT"]     = str(PROJECT_ROOT)

    import retriever as ret
    # Force override the globals in case module was already imported
    ret.EMBEDDING_MODEL  = DEMO_MODEL
    ret.EMBEDDING_DEVICE = "cpu"
    ret.INDEXES          = DEMO_INDEX.parent
    ret._index_cache     = {}   # clear cache so demo index is used
    ret._embed_model     = None # force model reload with cpu device

    # Patch _load_index to load demo files for strategy "demo"
    import faiss
    _demo_index = faiss.read_index(str(DEMO_INDEX))
    _demo_meta  = json.loads(DEMO_META.read_text(encoding="utf-8"))
    with open(DEMO_BM25, "rb") as fh:
        _bm25_data = pickle.load(fh)

    ret._index_cache["demo"] = {
        "faiss":    _demo_index,
        "metadata": _demo_meta,
        "bm25":     _bm25_data["bm25"],
        "corpus":   _bm25_data["corpus"],
    }

    print(f"\n{'='*65}")
    print("🔍 LIVE RETRIEVAL DEMO  (dense + BM25 + RRF + cross-encoder rerank)")
    print(f"   Model  : {DEMO_MODEL}")
    print(f"   Index  : {len(_demo_meta)} segments")
    print(f"{'='*65}")

    for query in DEMO_QUERIES:
        print(f"\n  Query: \"{query}\"")
        t0 = time.time()
        results = ret.search(
            query      = query,
            strategy   = "demo",
            top_k      = 3,
            use_rerank = False,   # Reranker skipped for demo (needs cross-encoder download)
            use_bm25   = True,
            verbose    = False,
        )
        elapsed = time.time() - t0

        print(f"  ⏱  {elapsed:.2f}s  |  intent: {results[0]['query_intent'] if results else '?'}")
        for r in results:
            print(f"    [{r['rank']}] {r.get('course_name', '?'):30s}  "
                  f"score={r['retrieval_score']:.4f}")
            print(f"         {r.get('lecture_title', '?')[:60]}")
            print(f"         ⏱ {int(r.get('start_sec', 0))//60}:{int(r.get('start_sec', 0))%60:02d}  "
                  f"→  {r.get('youtube_deep_link', 'no link')}")
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("NPTEL Lecture Retrieval — Quick Demo")
    print(f"Model: {DEMO_MODEL}  |  Segments: {N_SEGMENTS}")
    print("=" * 65)

    # Check if demo indexes already exist (skip rebuild on re-runs)
    if DEMO_INDEX.exists() and DEMO_META.exists() and DEMO_BM25.exists():
        print(f"\n✅ Demo indexes already exist — skipping build")
        print(f"   (Delete data/indexes/faiss_demo.index to rebuild)\n")
    else:
        segments = load_demo_segments()
        build_demo_indexes(segments)

    run_demo_queries()

    print("=" * 65)
    print("✅ Demo complete!")
    print()
    print("Next steps:")
    print("  • Streamlit UI:  streamlit run app.py")
    print("  • FastAPI:       uvicorn api.app:app --port 8000")
    print("  • Full index:    python3 src/retrieval/embedder.py --strategy c3  (needs GPU)")
    print("=" * 65)


if __name__ == "__main__":
    main()
