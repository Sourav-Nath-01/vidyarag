#!/usr/bin/env python3
"""
smoke_test.py — end-to-end check of a running VidyaRAG API (stdlib only)
=======================================================================
Waits for /ready, runs a real search, and checks input validation.
Used in three places:
  • CI, inside the freshly built container:  docker exec -i api python - <URL> < deploy/smoke_test.py
  • CD, against the live Azure URL right after a deploy
  • locally:  python deploy/smoke_test.py http://localhost:8000

Exit code 0 = healthy, 1 = failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def call(method: str, url: str, body: dict | None = None, timeout: float = 120):
    """Returns (status, json_body, headers); headers lookups are case-insensitive."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}"), resp.headers
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read() or b"{}")
        except ValueError:
            payload = {}
        return e.code, payload, e.headers


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base_url", help="e.g. http://127.0.0.1:7860 or https://<app>.azurecontainerapps.io")
    p.add_argument("--timeout", type=int, default=300, help="max seconds to wait for /ready")
    p.add_argument("--query", default="how does binary search tree insertion work")
    p.add_argument("--expect-version", default=None,
                   help="keep waiting until /ready reports this version (e.g. the git SHA just "
                        "deployed), so we test the NEW revision rather than the old one")
    args = p.parse_args()
    base = args.base_url.rstrip("/")

    # 1) Wait for readiness (covers scale-from-zero cold starts on Azure)
    t0 = time.time()
    last = "no response yet"
    while True:
        try:
            status, body, _ = call("GET", f"{base}/ready", timeout=60)
            version_ok = args.expect_version is None or body.get("version") == args.expect_version
            if status == 200 and version_ok:
                print(f"✓ ready after {time.time() - t0:.0f}s  {body}")
                break
            last = f"HTTP {status} {body}" + ("" if version_ok else f" (waiting for version {args.expect_version})")
        except Exception as e:  # connection refused, DNS, timeout while waking up...
            last = f"{type(e).__name__}: {e}"
        if time.time() - t0 > args.timeout:
            print(f"✗ not ready after {args.timeout}s — last: {last}")
            return 1
        time.sleep(5)

    # 2) Service info
    status, info, _ = call("GET", f"{base}/")
    print(f"✓ GET / → {status}  version={info.get('version')}  strategies={info.get('available_strategies')}")

    # 3) Real search through the whole pipeline
    status, res, headers = call("POST", f"{base}/search", {"query": args.query, "top_k": 3})
    if status != 200 or not res.get("results"):
        print(f"✗ search failed: HTTP {status} {res}")
        return 1
    top = res["results"][0]
    print(f"✓ POST /search → {res['n_results']} results in {res['latency_sec']}s "
          f"(server {headers.get('X-Response-Time-ms', '?')} ms)")
    print(f"  top hit: {top.get('course_name')} — {top.get('lecture_title')}  "
          f"score={top.get('retrieval_score'):.4f}")

    # 4) Validation still works
    status, _, _ = call("POST", f"{base}/search", {"query": "x"})
    if status != 422:
        print(f"✗ expected 422 for a too-short query, got {status}")
        return 1
    print("✓ invalid input rejected with 422")

    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
