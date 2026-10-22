"""Run the held-out test set against a served model and score it.

Reuses the schema check from registry/validate_dataset.py rather than
duplicating it, so Pillar 2 and Pillar 4 agree on what "valid" means.

Usage:
    export SERVER_URL=https://paulsilas938--cliniflow-vllm-serve.us-east.modal.run
    python evals/run_eval.py --transcripts-file training/colab/test.jsonl \
        --adapter-dir registry/adapters/config_A --out evals/results_finetuned.json

For the base model, point --server-url at the cliniflow-vllm-base deployment
and pass --model-name base-model (--adapter-dir is still used, only to load the
same system_prompt.txt so both models get an identical prompt).

Scoring: "valid" is strict (reply must be bare JSON); "lenient_valid" first strips
markdown fences / surrounding prose. Code metrics compare suggested_codes to the
gold output in the test file, using the lenient parse.
"""
import argparse
import json
import os
import re
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


def parse_lenient(text):
    """Pull a JSON object out of a reply that may be wrapped in markdown fences
    or surrounded by prose. Returns the parsed object, or None."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def norm_code(code):
    return re.sub(r"[^A-Z0-9]", "", str(code).upper())


def code_set(obj):
    codes = obj.get("suggested_codes") if isinstance(obj, dict) else None
    if not isinstance(codes, list):
        return set()
    return {norm_code(c["code"]) for c in codes if isinstance(c, dict) and c.get("code")}


def score_codes(pred_obj, gold_obj):
    """Gold-based ICD-10 check. Codes are compared after dropping punctuation/case."""
    gold, pred = code_set(gold_obj), code_set(pred_obj)
    hits = gold & pred
    return {
        "gold_codes": sorted(gold),
        "pred_codes": sorted(pred),
        "code_hit": bool(hits),
        "code_exact": gold == pred and bool(gold),
        "tp": len(hits), "n_pred": len(pred), "n_gold": len(gold),
    }


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
            problems = check_record(record)  # strict: reply must be bare JSON
            lenient_obj = parse_lenient(output_text)
            lenient_problems = (
                check_record({"input": row["input"], "output": lenient_obj})
                if lenient_obj is not None else ["no_json_found"]
            )
            gold = row["output"]
            if isinstance(gold, str):
                gold = json.loads(gold)
            results.append({
                "index": i,
                "valid": not problems,
                "problems": problems,
                "lenient_valid": not lenient_problems,
                "lenient_problems": lenient_problems,
                "codes": score_codes(lenient_obj, gold),
                "latency_s": round(latency, 2),
                "output": output_text,
            })
            print(f"[{i}/{len(rows)}] strict={not problems} lenient={not lenient_problems} latency={latency:.1f}s")
        except Exception as e:
            results.append({"index": i, "valid": False, "problems": [f"request_failed: {e}"],
                            "lenient_valid": False, "lenient_problems": [f"request_failed: {e}"],
                            "codes": None, "latency_s": None, "output": None})
            print(f"[{i}/{len(rows)}] FAILED: {e}")

    valid_count = sum(1 for r in results if r["valid"])
    lenient_count = sum(1 for r in results if r["lenient_valid"])
    latencies = [r["latency_s"] for r in results if r["latency_s"] is not None]
    scored = [r["codes"] for r in results if r["codes"]]
    tp = sum(c["tp"] for c in scored)
    n_pred = sum(c["n_pred"] for c in scored)
    n_gold = sum(c["n_gold"] for c in scored)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    n = len(rows)
    summary = {
        "rows": n,
        "valid": valid_count,
        "invalid": n - valid_count,
        "lenient_valid": lenient_count,
        "code_hit_rate": round(sum(c["code_hit"] for c in scored) / n, 3),
        "code_exact_rate": round(sum(c["code_exact"] for c in scored) / n, 3),
        "code_precision": round(precision, 3),
        "code_recall": round(recall, 3),
        "code_f1": round(2 * precision * recall / (precision + recall), 3) if tp else 0.0,
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n{summary}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
