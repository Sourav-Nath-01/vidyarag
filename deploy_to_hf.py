#!/usr/bin/env python3
"""
deploy_to_hf.py — One-command deploy to Hugging Face Spaces
============================================================
Creates a public Streamlit Space with the demo index.

Usage:
    source venv/bin/activate
    python deploy_to_hf.py --username YOUR_HF_USERNAME

What it does:
    1. Logs you into HF (opens browser)
    2. Creates Space: {username}/nptel-lecture-retrieval
    3. Uploads: app.py, src/, data/indexes/demo*, README_HF.md, requirements_hf.txt

The demo index (faiss_demo + bm25_demo + metadata_demo) is already built
in data/indexes/ — no GPU needed for deployment.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="SouravNath",
                        help="Your Hugging Face username (default: SouravNath)")
    parser.add_argument("--space-name", default="vidyarag",
                        help="Space name (default: vidyarag)")
    parser.add_argument("--token", default=None,
                        help="HF token (or set HF_TOKEN env var). If not set, will prompt login.")
    args = parser.parse_args()

    repo_id = f"{args.username}/{args.space_name}"
    print(f"\n{'='*60}")
    print(f"Deploying to: https://huggingface.co/spaces/{repo_id}")
    print(f"{'='*60}\n")

    try:
        from huggingface_hub import HfApi, login, create_repo
    except ImportError:
        print("huggingface-hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    # ── Login ────────────────────────────────────────────────────────────────
    import os
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("No token provided — opening browser for login...")
        login()

    api = HfApi()

    # ── Create Space ─────────────────────────────────────────────────────────
    print(f"\n[1/4] Creating Space: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
        print(f"  ✅ Space created (docker sdk)")
    except Exception as e:
        print(f"  ⚠️  Space creation: {e}")

    # Give HF a moment to provision the repo
    import time
    print("  Waiting 5s for Space to provision...")
    time.sleep(5)

    # ── Files to upload ──────────────────────────────────────────────────────
    print(f"\n[2/4] Uploading application files...")

    UPLOAD_FILES = [
        # App entry point (HF uses app.py by default for Streamlit)
        (PROJECT_ROOT / "app.py",              "app.py"),
        (PROJECT_ROOT / "requirements_hf.txt", "requirements.txt"),
        (PROJECT_ROOT / "README_HF.md",        "README.md"),
        (PROJECT_ROOT / ".env",                ".env"),
    ]

    UPLOAD_DIRS = [
        (PROJECT_ROOT / "src",          "src"),
        (PROJECT_ROOT / "configs",      "configs"),
    ]

    # Demo indexes only (not the full 500MB set)
    DEMO_INDEX_FILES = [
        "faiss_demo.index",
        "bm25_demo.pkl",
        "metadata_demo.json",
    ]

    for local_path, hf_path in UPLOAD_FILES:
        if local_path.exists():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=hf_path,
                repo_id=repo_id,
                repo_type="space",
            )
            print(f"  ✅ {hf_path}")
        else:
            print(f"  ⚠️  skipped (not found): {local_path}")

    for local_dir, hf_path in UPLOAD_DIRS:
        if local_dir.exists():
            api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=hf_path,
                repo_id=repo_id,
                repo_type="space",
                ignore_patterns=["__pycache__", "*.pyc", ".env"],
            )
            print(f"  ✅ {hf_path}/")

    # Demo indexes
    print(f"\n[3/4] Uploading demo indexes...")
    index_dir = PROJECT_ROOT / "data" / "indexes"
    for fname in DEMO_INDEX_FILES:
        fpath = index_dir / fname
        if fpath.exists():
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=f"data/indexes/{fname}",
                repo_id=repo_id,
                repo_type="space",
            )
            size_mb = fpath.stat().st_size / 1e6
            print(f"  ✅ data/indexes/{fname}  ({size_mb:.1f} MB)")
        else:
            print(f"  ❌ MISSING: {fpath}")

    print(f"\n[4/4] Done!")
    print(f"\n{'='*60}")
    print(f"🚀 Live at: https://huggingface.co/spaces/{repo_id}")
    print(f"   Build takes ~3-5 min after upload.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
