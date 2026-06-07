# ╔══════════════════════════════════════════════════════════════════╗
# ║  kaggle_upload_to_hf.py                                         ║
# ║  Paste this into a Kaggle notebook cell and run it.             ║
# ║  Uploads all FAISS/BM25/metadata indexes to HF Dataset repo.    ║
# ║  Kaggle internet: ~1GB/s → 560MB uploads in ~5 seconds.         ║
# ╚══════════════════════════════════════════════════════════════════╝

import os, glob, subprocess, sys
from pathlib import Path

# ── Step 1: Install HF hub if not present ────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"], check=True)

from huggingface_hub import HfApi, login, create_repo

# ── Step 2: Login ────────────────────────────────────────────────────────────
# PASTE YOUR NEW HF TOKEN BELOW (get from huggingface.co/settings/tokens)
HF_TOKEN  = "YOUR_HF_TOKEN_HERE"   # ← replace this
REPO_ID   = "SouravNath/vidyarag-indexes"
REPO_TYPE = "dataset"

login(token=HF_TOKEN)
api = HfApi()

# Create dataset repo if not exists
try:
    create_repo(REPO_ID, repo_type=REPO_TYPE, private=False, exist_ok=True)
    print(f"✅ Repo ready: {REPO_ID}")
except Exception as e:
    print(f"ℹ️  {e}")

# ── Step 3: Find index files on Kaggle ───────────────────────────────────────
# Check common Kaggle paths where indexes might be
SEARCH_PATHS = [
    "/kaggle/working/data/indexes",
    "/kaggle/working/nptel-lecture-retrieval-main/data/indexes",
    "/kaggle/working/indexes",
    "/kaggle/input/nptel-indexes",
    "/kaggle/input",
]

INDEX_PATTERNS = [
    "faiss_c1.index", "faiss_c2.index", "faiss_c3.index",
    "faiss_c2_w150.index", "faiss_c2_w250.index",
    "faiss_c3_t025.index", "faiss_c3_t040.index",
    "bm25_c1.pkl", "bm25_c2.pkl", "bm25_c3.pkl",
    "bm25_c2_w150.pkl", "bm25_c2_w250.pkl",
    "bm25_c3_t025.pkl", "bm25_c3_t040.pkl",
    "metadata_c1.json", "metadata_c2.json", "metadata_c3.json",
    "metadata_c2_w150.json", "metadata_c2_w250.json",
    "metadata_c3_t025.json", "metadata_c3_t040.json",
]

found = {}
for search_dir in SEARCH_PATHS:
    p = Path(search_dir)
    if p.exists():
        print(f"\n📂 Searching {search_dir}...")
        for pattern in INDEX_PATTERNS:
            # Also search recursively
            matches = list(p.glob(f"**/{pattern}")) + [p / pattern]
            for m in matches:
                if m.exists() and m.name not in found:
                    found[m.name] = m
                    print(f"   Found: {m.name}  ({m.stat().st_size/1e6:.1f} MB)")

if not found:
    # Last resort: find all .index and .pkl files
    print("\n⚠️  Searching all of /kaggle/working/ ...")
    for ext in ["*.index", "*.pkl", "metadata_c*.json"]:
        for f in Path("/kaggle/working").glob(f"**/{ext}"):
            if f.name not in found and any(pat in f.name for pat in ["c1","c2","c3"]):
                found[f.name] = f
                print(f"   Found: {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

print(f"\n{'='*50}")
print(f"Total files found: {len(found)}")
print(f"{'='*50}")

if not found:
    print("❌ No index files found. Make sure indexes are built in this Kaggle session.")
    print("   Run the index builder first: python build_indexes.py")
else:
    # ── Step 4: Upload to HF Dataset ─────────────────────────────────────────
    import time
    total = len(found)
    for i, (fname, fpath) in enumerate(sorted(found.items()), 1):
        size_mb = fpath.stat().st_size / 1e6
        print(f"\n[{i}/{total}] Uploading {fname} ({size_mb:.1f} MB)...", flush=True)
        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"upload: {fname}",
            )
            elapsed = time.time() - t0
            speed   = size_mb / elapsed if elapsed > 0 else 0
            print(f"   ✅ Done in {elapsed:.1f}s  ({speed:.1f} MB/s)")
        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print(f"\n{'='*50}")
    print("🎉 All uploads complete!")
    print(f"Dataset: https://huggingface.co/datasets/{REPO_ID}")
    print(f"{'='*50}")
