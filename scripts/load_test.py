"""Load test: drive /ingest concurrently, then verify every transaction landed.

    python -m scripts.load_test --total 5000 --concurrency 32

Black-box on the way in (HTTP only), white-box on the way out: it checks
Postgres directly that every transaction the API acknowledged was scored and
persisted, which is the loss check that matters.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import psycopg

from src import config
from src.model.preprocess import FEATURE_ORDER


def _post(url: str, features: list[float]) -> tuple[str | None, float, int]:
    body = json.dumps({"features": features}).encode()
    req = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            tid = json.loads(resp.read())["transaction_id"]
            return tid, time.perf_counter() - started, resp.status
    except urllib.error.HTTPError as exc:
        return None, time.perf_counter() - started, exc.code
    except OSError:
        return None, time.perf_counter() - started, 0


def _persisted(ids: list[str]) -> int:
    with psycopg.connect(config.DATABASE_URL) as conn:
        return conn.execute(
            "SELECT count(*) FROM fraud_scores WHERE transaction_id = ANY(%s)", (ids,)
        ).fetchone()[0]


def _pct(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=100)[q - 1] if len(values) > 1 else values[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000/ingest")
    parser.add_argument("--total", type=int, default=5000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--drain-timeout", type=float, default=120)
    args = parser.parse_args()

    rows = pd.read_csv("data/creditcard.csv")[FEATURE_ORDER].values.tolist()
    sample = random.Random(0).choices(rows, k=args.total)

    print(f"Sending {args.total} transactions with concurrency {args.concurrency}...")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda f: _post(args.url, f), sample))
    ingest_seconds = time.perf_counter() - started

    ids = [tid for tid, _, _ in results if tid]
    failures: dict[int, int] = {}
    for tid, _, status in results:
        if not tid:
            failures[status] = failures.get(status, 0) + 1
    latencies = [lat * 1000 for tid, lat, _ in results if tid]

    print(f"  ingest: {len(ids)}/{args.total} acked in {ingest_seconds:.1f}s "
          f"({len(ids) / ingest_seconds:.0f} req/s)")
    if latencies:
        print(f"  latency ms: p50 {_pct(latencies, 50):.0f}  p95 {_pct(latencies, 95):.0f}  "
              f"p99 {_pct(latencies, 99):.0f}  max {max(latencies):.0f}")
    if failures:
        print(f"  failures by status: {failures}")

    print("Waiting for every acked transaction to be scored and persisted...")
    deadline = time.monotonic() + args.drain_timeout
    while True:
        done = _persisted(ids)
        if done == len(ids) or time.monotonic() > deadline:
            break
        time.sleep(1)
    total_seconds = time.perf_counter() - started

    missing = len(ids) - done
    print(f"  persisted: {done}/{len(ids)} in {total_seconds:.1f}s from first request "
          f"({done / total_seconds:.0f} end-to-end/s)")
    print("  RESULT:", "no loss" if missing == 0 else f"{missing} MISSING")
    raise SystemExit(0 if missing == 0 else 1)


if __name__ == "__main__":
    main()
