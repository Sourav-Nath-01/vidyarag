#!/usr/bin/env python3
"""
api_space.py — One-command deploy of the FastAPI REST API to a
Docker-based Hugging Face Space (free CPU tier).

Usage:
    export HF_TOKEN=hf_...          # write-access token
    python deploy/api_space.py --username YOUR_HF_USERNAME

What it does:
    1. Creates a Docker-SDK Space: {username}/vidyarag-api
    2. Uploads: deploy/Dockerfile, deploy/requirements/api.txt, api/, src/,
       configs/, deploy/space_cards/api.md
    3. The Space's own bootstrap logic (in api/app.py) downloads the full
       BGE-large indexes from the SouravNath/vidyarag-indexes dataset on
       first container start — no index files are uploaded here.

The Streamlit demo Space is deployed separately with deploy/streamlit_space.py.
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="SouravNath",
                        help="Your Hugging Face username (default: SouravNath)")
    parser.add_argument("--space-name", default="vidyarag-api",
                        help="Space name (default: vidyarag-api)")
    parser.add_argument("--token", default=None,
                        help="HF token (or set HF_TOKEN env var). If not set, will prompt login.")
    args = parser.parse_args()

    repo_id = f"{args.username}/{args.space_name}"
    print(f"\n{'='*60}")
    print(f"Deploying API to: https://huggingface.co/spaces/{repo_id}")
    print(f"{'='*60}\n")

    try:
        from huggingface_hub import HfApi, login, create_repo
    except ImportError:
        print("huggingface-hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("No token provided — opening browser for login...")
        login()

    api = HfApi()

    print(f"\n[1/3] Creating Space: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            exist_ok=True,
        )
        print("  ✅ Space created (docker sdk)")
    except Exception as e:
        print(f"  ⚠️  Space creation: {e}")

    print("  Waiting 5s for Space to provision...")
    time.sleep(5)

    print("\n[2/3] Uploading application files...")

    UPLOAD_FILES = [
        (PROJECT_ROOT / "deploy/Dockerfile",                 "Dockerfile"),
        (PROJECT_ROOT / "deploy/requirements/api.txt",       "deploy/requirements/api.txt"),
        (PROJECT_ROOT / "deploy/space_cards/api.md",         "README.md"),
    ]
    UPLOAD_DIRS = [
        (PROJECT_ROOT / "api",     "api"),
        (PROJECT_ROOT / "src",     "src"),
        (PROJECT_ROOT / "configs", "configs"),
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

    print("\n[3/3] Done!")
    print(f"\n{'='*60}")
    print(f"\U0001f680 Building at: https://huggingface.co/spaces/{repo_id}")
    print("   Docker build + first-boot index download takes ~3-6 min.")
    print(f"   API docs will be at: https://huggingface.co/spaces/{repo_id} -> /docs")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
