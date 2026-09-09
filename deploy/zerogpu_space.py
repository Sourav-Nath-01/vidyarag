#!/usr/bin/env python3
"""
zerogpu_space.py — One-command deploy of the GPU-accelerated search
API to a free Gradio + ZeroGPU Hugging Face Space.

Usage:
    export HF_TOKEN=hf_...          # write-access token
    python deploy/zerogpu_space.py --username YOUR_HF_USERNAME

Why ZeroGPU instead of a Docker Space: as of 2026, HF requires a PRO
subscription to create new Gradio/Docker Spaces on cpu-basic hardware.
Gradio Spaces on ZeroGPU hardware remain free for personal accounts
(up to 2), with a 3.5 GPU-minute/day quota. See deploy/space_cards/zerogpu.md.
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", default="SouravNath")
    parser.add_argument("--space-name", default="vidyarag-api")
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    repo_id = f"{args.username}/{args.space_name}"
    print(f"\n{'='*60}")
    print(f"Deploying ZeroGPU API to: https://huggingface.co/spaces/{repo_id}")
    print(f"{'='*60}\n")

    try:
        from huggingface_hub import HfApi, login, create_repo, SpaceHardware
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

    print(f"\n[1/3] Creating Space: {repo_id} (gradio sdk, ZeroGPU hardware)")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=SpaceHardware.ZERO_A10G,
            private=False,
            exist_ok=True,
        )
        print("  ✅ Space created (gradio sdk, zero-a10g hardware)")
    except Exception as e:
        print(f"  ⚠️  Space creation: {e}")

    print("  Waiting 5s for Space to provision...")
    time.sleep(5)

    print("\n[2/3] Uploading application files...")

    UPLOAD_FILES = [
        (PROJECT_ROOT / "deploy/app_zerogpu.py",              "app.py"),
        (PROJECT_ROOT / "deploy/requirements/zerogpu.txt",    "requirements.txt"),
        (PROJECT_ROOT / "deploy/space_cards/zerogpu.md",      "README.md"),
    ]
    UPLOAD_DIRS = [
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
    print("   First boot: index download (~500MB) + model warmup, ~3-6 min.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
