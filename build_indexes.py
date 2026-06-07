"""
build_indexes.py  —  One-shot index builder (CPU-friendly)
===========================================================
Builds FAISS dense + BM25 sparse indexes for ALL strategies:
  C1, C2, C3, C2_w150, C2_w250, C3_t025, C3_t040

Uses all-MiniLM-L6-v2 (384-dim, ~90MB) which runs at ~500 segs/sec on CPU.
Total time on 8-core CPU: ~3–5 minutes for all 51k segments.

When GPU is available, switch to BGE-large by running:
    PROJECT_ROOT=. EMBEDDING_MODEL=BAAI/bge-large-en-v1.5 EMBEDDING_DEVICE=cuda \\
    python src/retrieval/embedder.py

Usage:
    source venv/bin/activate
    python build_indexes.py                  # build all strategies
    python build_indexes.py --strategy c3   # one strategy only
    python build_indexes.py --bm25-only     # only BM25 (skip FAISS)
    python build_indexes.py --dry-run       # print plan, no files written
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
INDEXES      = PROJECT_ROOT / "data" / "indexes"
sys.path.insert(0, str(PROJECT_ROOT / "src" / "retrieval"))

# Load .env if present
try:
    from dotenv import load_dotenv
    if (PROJECT_ROOT / ".env").exists():
        load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

# ── model config ──────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",  "all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
BATCH_SIZE       = int(os.getenv("EMBEDDING_BATCH_SIZE", "128"))

# ── all strategies to build ───────────────────────────────────────────────────
ALL_STRATEGIES = {
    "c1":      "segments_c1.jsonl",
    "c2":      "segments_c2.jsonl",
    "c3":      "segments_c3.jsonl",
    "c2_w150": "segments_c2_w150.jsonl",
    "c2_w250": "segments_c2_w250.jsonl",
    "c3_t025": "segments_c3_t025.jsonl",
    "c3_t040": "segments_c3_t040.jsonl",
}

# ── stopwords (must match retriever.py exactly) ───────────────────────────────
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "of",
    "is", "it", "as", "be", "by", "we", "so", "do", "if", "he", "she",
    "this", "that", "with", "from", "are", "was", "were", "has", "have",
    "had", "not", "also", "will", "can", "its", "our", "their", "what",
    "which", "when", "there", "then", "they", "them", "been", "more",
    "into", "than", "just", "some", "would", "about", "because", "now",
    "very", "here", "like", "okay", "right", "yeah", "uh", "um",
}


def _tokenise(text: str) -> list[str]:
    text   = text.lower()
    text   = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    return [t for t in tokens if len(t) >= 2 and t not in _STOPWORDS]


# ─────────────────────────────────────────────────────────────────────────────
# Shared: load segments from .jsonl
# ─────────────────────────────────────────────────────────────────────────────

def load_segments(jsonl_path: Path) -> list[dict]:
    segments = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    segments.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return segments


# ─────────────────────────────────────────────────────────────────────────────
# FAISS index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_passage_text(seg: dict) -> str:
    """Builds the text passage for dense embedding (same as embedder.py)."""
    parts = ["passage:"]

    course = seg.get("course_name", "")
    if course:
        parts.append(f"[COURSE]: {course}")

    lecture = seg.get("lecture_title", "")
    if lecture:
        parts.append(f"[LECTURE]: {lecture}")

    ctype = seg.get("content_type", "")
    if ctype:
        parts.append(f"[TYPE]: {ctype}")

    ocr = seg.get("ocr_text", "").strip()
    if ocr and not seg.get("ocr_failed", False):
        ocr_clean = ocr.replace("\n---\n", " ").replace("\n", " ").strip()
        if len(ocr_clean) > 10:
            parts.append(f"[SLIDE]: {ocr_clean}")

    transcript = seg.get("transcript", "").strip()
    if transcript:
        parts.append(f"[SPEECH]: {transcript}")

    return " ".join(parts)


def build_faiss_index(strategy: str, jsonl_path: Path, model, dry_run: bool) -> None:
    import numpy as np
    import faiss

    faiss_out = INDEXES / f"faiss_{strategy}.index"
    meta_out  = INDEXES / f"metadata_{strategy}.json"

    print(f"\n{'='*60}", flush=True)
    print(f"[FAISS] Strategy: {strategy.upper()} | {jsonl_path.name}", flush=True)

    # Skip if already built (unless forced)
    if faiss_out.exists():
        print(f"  ✅ Already exists: {faiss_out.name}  (skipping)", flush=True)
        return

    if not jsonl_path.exists():
        print(f"  ⚠️  SKIP — file not found: {jsonl_path}", flush=True)
        return

    segments = load_segments(jsonl_path)
    print(f"  Loaded {len(segments):,} segments", flush=True)

    texts = [build_passage_text(seg) for seg in segments]

    if dry_run:
        print(f"  [DRY RUN] Would embed {len(texts):,} passages with {EMBEDDING_MODEL}", flush=True)
        return

    # Embed
    print(f"  Embedding with {EMBEDDING_MODEL} on {EMBEDDING_DEVICE} "
          f"(batch={BATCH_SIZE}) ...", flush=True)
    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  ({len(texts)/elapsed:.0f} segs/sec)", flush=True)

    # Build FAISS index
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"  FAISS: {index.ntotal:,} vectors, dim={dim}", flush=True)

    # Save FAISS
    INDEXES.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(faiss_out))
    print(f"  Saved → {faiss_out.name}  ({faiss_out.stat().st_size/1e6:.1f} MB)", flush=True)

    # Save metadata (existing metadata_*.json is already present but may differ from
    # what embedder.py writes — reuse existing to avoid 40MB rewrite if it matches)
    if not meta_out.exists():
        meta_fields = [
            "segment_id", "course_id", "course_name", "instructor", "institute",
            "lecture_title", "lecture_number", "youtube_url", "youtube_deep_link",
            "start_sec", "end_sec", "duration_sec",
            "transcript", "ocr_text", "ocr_confidence", "ocr_failed",
            "is_code_segment", "content_type", "chunking_strategy",
            "word_count", "boundary_method",
        ]
        slim = [{k: seg.get(k) for k in meta_fields} for seg in segments]
        meta_out.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
        print(f"  Saved → {meta_out.name}", flush=True)
    else:
        print(f"  ✅ Metadata already exists: {meta_out.name}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# BM25 index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_bm25_text(seg: dict) -> str:
    """Builds text for BM25 tokenisation (matches bm25_builder.py)."""
    parts = []

    course = seg.get("course_name", "")
    if course:
        parts.append(course)

    ocr = seg.get("ocr_text", "").strip()
    if ocr and not seg.get("ocr_failed", False):
        ocr_clean = ocr.replace("\n---\n", " ").replace("\n", " ").strip()
        if len(ocr_clean) > 10:
            parts.append(ocr_clean)
            parts.append(ocr_clean)  # ×2 weight for slide text

    transcript = seg.get("transcript", "").strip()
    if transcript:
        parts.append(transcript)

    return " ".join(parts)


def build_bm25_index(strategy: str, jsonl_path: Path, dry_run: bool) -> None:
    from rank_bm25 import BM25Okapi

    out_path = INDEXES / f"bm25_{strategy}.pkl"

    print(f"\n{'='*60}", flush=True)
    print(f"[BM25]  Strategy: {strategy.upper()} | {jsonl_path.name}", flush=True)

    if out_path.exists():
        print(f"  ✅ Already exists: {out_path.name}  (skipping)", flush=True)
        return

    if not jsonl_path.exists():
        print(f"  ⚠️  SKIP — file not found: {jsonl_path}", flush=True)
        return

    segments = load_segments(jsonl_path)
    print(f"  Loaded {len(segments):,} segments", flush=True)

    if dry_run:
        print(f"  [DRY RUN] Would tokenise {len(segments):,} segments", flush=True)
        return

    t0     = time.time()
    corpus = [_tokenise(build_bm25_text(seg)) for seg in segments]
    bm25   = BM25Okapi(corpus)
    elapsed = time.time() - t0

    INDEXES.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        pickle.dump({"bm25": bm25, "corpus": corpus}, fh, protocol=pickle.HIGHEST_PROTOCOL)

    vocab_size  = len(bm25.idf)
    avg_doc_len = sum(len(d) for d in corpus) / len(corpus)
    size_mb     = out_path.stat().st_size / 1e6

    print(f"  Done in {elapsed:.1f}s | vocab={vocab_size:,} | avg_len={avg_doc_len:.1f}", flush=True)
    print(f"  Saved → {out_path.name}  ({size_mb:.1f} MB)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build FAISS + BM25 indexes for all strategies"
    )
    parser.add_argument("--strategy",   default=None, choices=list(ALL_STRATEGIES.keys()),
                        help="Build only this strategy (default: all)")
    parser.add_argument("--bm25-only",  action="store_true",
                        help="Skip FAISS, only build BM25 indexes")
    parser.add_argument("--faiss-only", action="store_true",
                        help="Skip BM25, only build FAISS indexes")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print plan without writing any files")
    parser.add_argument("--force",      action="store_true",
                        help="Rebuild even if index already exists")
    args = parser.parse_args()

    strategies = (
        {args.strategy: ALL_STRATEGIES[args.strategy]}
        if args.strategy
        else ALL_STRATEGIES
    )

    print("=" * 60, flush=True)
    print("NPTEL Index Builder", flush=True)
    print(f"  Model   : {EMBEDDING_MODEL}", flush=True)
    print(f"  Device  : {EMBEDDING_DEVICE}", flush=True)
    print(f"  Batch   : {BATCH_SIZE}", flush=True)
    print(f"  Targets : {list(strategies.keys())}", flush=True)
    print(f"  Dry run : {args.dry_run}", flush=True)
    print("=" * 60, flush=True)

    # ── If force, remove existing indexes so they get rebuilt ─────────────────
    if args.force:
        for sid in strategies:
            for p in [INDEXES / f"faiss_{sid}.index", INDEXES / f"bm25_{sid}.pkl"]:
                if p.exists():
                    p.unlink()
                    print(f"  [FORCE] Removed {p.name}", flush=True)

    # ── BM25 (fast, CPU, no model needed) ────────────────────────────────────
    if not args.faiss_only:
        print("\n\n── Building BM25 indexes ─────────────────────────────────", flush=True)
        t_bm25 = time.time()
        for sid, fname in strategies.items():
            build_bm25_index(sid, PROCESSED / fname, args.dry_run)
        print(f"\n✅ BM25 total: {time.time()-t_bm25:.1f}s", flush=True)

    # ── FAISS (needs embedding model) ─────────────────────────────────────────
    if not args.bm25_only:
        print("\n\n── Building FAISS indexes ────────────────────────────────", flush=True)
        print(f"Loading model: {EMBEDDING_MODEL} on {EMBEDDING_DEVICE} ...", flush=True)

        if not args.dry_run:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
            print(f"Model loaded. Embedding dim = {model.get_sentence_embedding_dimension()}\n",
                  flush=True)
        else:
            model = None

        t_faiss = time.time()
        for sid, fname in strategies.items():
            build_faiss_index(sid, PROCESSED / fname, model, args.dry_run)
        print(f"\n✅ FAISS total: {time.time()-t_faiss:.1f}s", flush=True)

    print("\n\n" + "=" * 60, flush=True)
    print("✅ All done! Index summary:", flush=True)
    for sid in strategies:
        fi = INDEXES / f"faiss_{sid}.index"
        bi = INDEXES / f"bm25_{sid}.pkl"
        fi_str = f"{fi.stat().st_size/1e6:.0f}MB" if fi.exists() else "MISSING"
        bi_str = f"{bi.stat().st_size/1e6:.0f}MB" if bi.exists() else "MISSING"
        print(f"  {sid:<10}  faiss={fi_str:<10} bm25={bi_str}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
