---
title: NPTEL Lecture Retrieval API
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# NPTEL Lecture Retrieval — FastAPI

**REST API for multimodal NPTEL lecture search.**  
Achieves **MRR 0.826 / Recall@10 0.964** on 84 human-annotated queries.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check + model info |
| `POST` | `/search` | Run full retrieval pipeline |
| `GET` | `/strategies` | List available chunking strategies |
| `GET` | `/docs` | Interactive Swagger UI |

## Example request

```bash
curl -X POST https://<your-space>.hf.space/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how does binary search tree insertion work",
    "strategy": "c3",
    "top_k": 5,
    "use_rerank": true,
    "use_bm25": true
  }'
```

## Response format

```json
{
  "query": "...",
  "strategy": "c3",
  "latency_sec": 0.42,
  "n_results": 5,
  "results": [
    {
      "rank": 1,
      "course_name": "Introduction to Algorithms and Analysis",
      "lecture_title": "Binary Search Trees",
      "start_sec": 415.36,
      "youtube_deep_link": "https://youtube.com/watch?v=...&t=415s",
      "retrieval_score": 0.031245,
      "query_intent": "procedural"
    }
  ]
}
```

## Pipeline

```
Query → BGE-large embed → FAISS top-100 → BM25 top-100 → RRF → boost → cross-encoder rerank → dedup → results
```
