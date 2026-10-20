"""Run the held-out test set against a served model and score it.

Reuses the schema check from registry/validate_dataset.py rather than
duplicating it, so Pillar 2 and Pillar 4 agree on what "valid" means.

Usage:
    export SERVER_URL=https://paulsilas938--cliniflow-vllm-serve.us-east.modal.run
    python evals/run_eval.py --transcripts-file training/colab/test.jsonl \
        --adapter-dir registry/adapters/config_A --out evals/results_finetuned.json

For the base model (no adapter), point --server-url at a server started
without --enable-lora and the "model" field in requests will just be the
base model's own name - see evals/README.md once the base-model run exists.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "registry"))
sys.path.insert(0, os.path.join(ROOT, "serving"))

from validate_dataset import check_record  # noqa: E402


def load_system_prompt(adapter_dir):
    path = os.path.join(adapter_dir, "system_prompt.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_model(server_url, model_name, system_prompt, transcript, timeout=600):
    body = json.dumps({
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ],
        "max_tokens": 1500,
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{server_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    latency = time.perf_counter() - start
    return payload["choices"][0]["message"]["content"].strip(), latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcripts-file", required=True)
    parser.add_argument("--server-url", default=os.environ.get("SERVER_URL", "http://localhost:8000"))
    parser.add_argument("--model-name", default="clinical-json-extractor")
    parser.add_argument("--adapter-dir", required=True, help="for system_prompt.txt")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    system_prompt = load_system_prompt(args.adapter_dir)

    rows = []
    with open(args.transcripts_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    results = []
    for i, row in enumerate(rows, 1):
        try:
            output_text, latency = call_model(args.server_url, args.model_name, system_prompt, row["input"])
            record = {"input": row["input"], "output": output_text}
            problems = check_record(record)
            results.append({
                "index": i,
                "valid": not problems,
                "problems": problems,
                "latency_s": round(latency, 2),
                "output": output_text,
            })
            print(f"[{i}/{len(rows)}] valid={not problems} latency={latency:.1f}s")
        except Exception as e:
            results.append({"index": i, "valid": False, "problems": [f"request_failed: {e}"], "latency_s": None, "output": None})
            print(f"[{i}/{len(rows)}] FAILED: {e}")

    valid_count = sum(1 for r in results if r["valid"])
    latencies = [r["latency_s"] for r in results if r["latency_s"] is not None]
    summary = {
        "rows": len(rows),
        "valid": valid_count,
        "invalid": len(rows) - valid_count,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n{summary}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
