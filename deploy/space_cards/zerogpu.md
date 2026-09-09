---
title: VidyaRAG API (ZeroGPU)
emoji: ⚡
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
license: mit
---

# VidyaRAG — GPU-accelerated Search API (ZeroGPU)

**Free, on-demand GPU inference** for the same hybrid retrieval pipeline as the
main [VidyaRAG demo](https://huggingface.co/spaces/SouravNath/vidyarag) —
BGE-large + BM25 + RRF + cross-encoder — running on Hugging Face's ZeroGPU
(dynamically allocated A10G, released after each request).

## Quota

HF's free tier grants **3.5 GPU-minutes/day** per account. Each search here is
budgeted at 25s, so this Space supports roughly **8 searches/day** before
quota resets 24h after first use. That's the deliberate tradeoff for genuinely
free GPU inference — for unlimited access, use the always-on CPU Streamlit
demo instead.

## Try it

Use the search box above, or call the REST API directly:

```bash
curl -X POST https://souravnath-vidyarag-api.hf.space/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how does binary search tree insertion work", "strategy": "c3", "top_k": 5}'
```

Interactive API docs: `/docs`

## Why two Spaces?

| | Streamlit demo | This Space (ZeroGPU) |
|---|---|---|
| Hardware | CPU Basic (free, unlimited) | A10G GPU (free, ~8 req/day) |
| Purpose | Primary, always-on demo | GPU-accelerated REST API showcase |
| Latency | ~5-8s/query | Sub-second once warm |
