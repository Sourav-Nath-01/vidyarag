---
title: VidyaRAG - NPTEL Lecture Retrieval
emoji: 🎓
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
license: mit
---

# VidyaRAG — Multimodal Hybrid Lecture Retrieval

**Multimodal semantic search over NPTEL lecture videos.**  
Achieves **MRR 0.826 / Recall@10 0.964** on 84 human-annotated queries across 9 CS courses.

## How to use

1. Select a chunking strategy from the sidebar (`Demo` is fastest — runs on CPU with no wait)
2. Type a natural language query in the search box
3. Click **Search** — results include YouTube deep-links that jump to the exact lecture timestamp

## Strategies

| Strategy | Description | Speed |
|---|---|---|
| **Demo** | 300-segment mini-index (MiniLM) | Instant on CPU |
| **C1** | Fixed 30s windows | Needs full index |
| **C2** | Word-count boundaries (~200w) | Needs full index |
| **C3** | Slide-boundary via OCR Jaccard (best) | Needs full index |

## Example queries

- `how does binary search tree insertion work`
- `explain backpropagation in neural networks`
- `what is virtual memory and paging`
- `SQL GROUP BY HAVING aggregate functions`
- `explain attention mechanism in sequence models`

## Tech stack

`BGE-large-en-v1.5` · `FAISS` · `BM25` · `RRF` · `cross-encoder/ms-marco-MiniLM-L-6-v2` · `Whisper ASR` · `Tesseract OCR` · `Ollama`
