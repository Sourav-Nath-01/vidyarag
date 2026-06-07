"""
app_hf.py — HuggingFace Spaces entry point
==========================================
Wraps app.py for HF Spaces deployment.
Forces CPU device and demo strategy as default.
Demo index (300 segments, MiniLM 384-dim) loads in ~2s on CPU.
"""
import os

# ── Force CPU for HF Spaces free tier ─────────────────────────────────────────
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("EMBEDDING_MODEL",  "all-MiniLM-L6-v2")
os.environ.setdefault("DEFAULT_STRATEGY", "demo")
os.environ.setdefault("PROJECT_ROOT",     os.path.dirname(os.path.abspath(__file__)))

# Run the main app — all logic is in app.py
exec(open(os.path.join(os.path.dirname(__file__), "app.py")).read())
