"""Load test: send N transcripts at C concurrent requests and record speed.

Measures:
  - time to first token (TTFT): how long until the first piece of the reply arrives
  - total latency per request
  - throughput: generated tokens per second across the whole run (approximate:
    counts streamed chunks, which is close to tokens for this model)

Standard library only.

Usage:
    python serving/load_test.py --transcripts-file training/colab/val.jsonl --n 20 --concurrency 5
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config  # noqa: E402
from client import build_request, load_system_prompt  # noqa: E402


def stream_one(transcript, system_prompt):
    body = json.dumps({**build_request(transcript, system_prompt), "stream": True}).encode("utf-8")
    req = urllib.request.Request(
        f"{config.SERVER_URL}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    ttft = None
    chunks = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            delta = json.loads(line[5:])["choices"][0]["delta"].get("content")
            if delta:
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
    total = time.perf_counter() - start
    return {"ttft": ttft or total, "total": total, "tokens": chunks}


def percentile(values, p):
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1))))
    return ordered[k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts-file", required=True, help="JSONL with an 'input' field per line")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    args = parser.parse_args()

    transcripts = []
    with open(args.transcripts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                transcripts.append(json.loads(line)["input"])
            if len(transcripts) >= args.n:
                break

    system_prompt = load_system_prompt()
    run_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda t: stream_one(t, system_prompt), transcripts))
    wall = time.perf_counter() - run_start

    ttfts = [r["ttft"] for r in results]
    totals = [r["total"] for r in results]
    tokens = sum(r["tokens"] for r in results)

    print(f"requests={len(results)} concurrency={args.concurrency} wall={wall:.1f}s")
    print(f"TTFT   median={statistics.median(ttfts):.2f}s  p95={percentile(ttfts, 95):.2f}s")
    print(f"Latency median={statistics.median(totals):.2f}s  p95={percentile(totals, 95):.2f}s")
    print(f"Throughput ~{tokens / wall:.1f} tokens/s (approx, aggregate)")


if __name__ == "__main__":
    main()
