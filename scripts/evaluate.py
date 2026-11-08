"""Lightweight regression eval against a running API (no LLM judge, no extra cost).

    python scripts/evaluate.py eval/golden.example.jsonl --api http://localhost:8000

Each JSONL line: {"question": "...", "expected_source": "file.md", "keywords": ["a", "b"]}
Reports: retrieval hit-rate (expected source among the sources used), keyword recall, grounded rate, latency p50/p95.
"""
import argparse
import json
import os
import statistics
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("golden")
    ap.add_argument("--api", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    ap.add_argument("--min-hit-rate", type=float, default=0.0, help="exit 1 if retrieval hit-rate is below this")
    args = ap.parse_args()
    headers = {"X-API-Key": os.environ["API_AUTH_KEY"]} if os.getenv("API_AUTH_KEY") else {}
    rows = [json.loads(line) for line in open(args.golden, encoding="utf-8") if line.strip()]

    hits = kw_scores = grounded = 0
    lat: list[float] = []
    for i, row in enumerate(rows, 1):
        t = time.perf_counter()
        r = httpx.post(f"{args.api}/v1/chat", json={"message": row["question"], "use_cache": False}, headers=headers, timeout=120)
        lat.append(time.perf_counter() - t)
        if r.status_code != 200:
            print(f"[{i}] HTTP {r.status_code}: {r.text[:120]}")
            continue
        body = r.json()
        src_hit = any(s["location"].endswith(row["expected_source"]) for s in body["sources"]) if row.get("expected_source") else True
        kws = row.get("keywords", [])
        kw = sum(k.lower() in body["answer"].lower() for k in kws) / len(kws) if kws else 1.0
        hits += src_hit; kw_scores += kw; grounded += body["grounded"]
        print(f"[{i}] source={'ok' if src_hit else 'MISS'} keywords={kw:.0%} provider={body['provider']} {lat[-1]:.1f}s  {row['question'][:60]}")
    n = len(rows)
    lat.sort()
    print(f"\nretrieval hit-rate {hits / n:.0%} | keyword recall {kw_scores / n:.0%} | grounded {grounded / n:.0%} | "
          f"latency p50 {statistics.median(lat):.1f}s p95 {lat[min(n - 1, int(n * 0.95))]:.1f}s")
    return 1 if hits / n < args.min_hit_rate else 0


if __name__ == "__main__":
    sys.exit(main())
