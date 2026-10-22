"""Offline rescoring of run_eval.py result files: no server calls.

Adds what the original runs did not record: category-level code matching
(first 3 characters of the ICD-10 code), medians, latency normalised by output
length, and a view that excludes rows that failed for network reasons.

Usage:
    python evals/rescore.py --test training/colab/test.jsonl \
        --base evals/results_base.json --finetuned evals/results_finetuned.json \
        --out evals/rescored.json
"""
import argparse
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import code_set, parse_lenient  # noqa: E402

ICD10_SHAPE = re.compile(r"^[A-Z]\d{2}[A-Z0-9]{0,4}$")  # dot-less ICD-10-CM


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_obj(row):
    g = row["output"]
    return json.loads(g) if isinstance(g, str) else g


def descriptions(obj):
    codes = obj.get("suggested_codes") if isinstance(obj, dict) else None
    if not isinstance(codes, list):
        return set()
    return {re.sub(r"[^A-Z0-9]", "", str(c["description"]).upper())
            for c in codes if isinstance(c, dict) and c.get("description")}


def categories(codes):
    return {c[:3] for c in codes}


def score(results, gold_rows, label):
    n_all = len(results)
    ok = [r for r in results if r["latency_s"] is not None]  # drop request failures
    per_row = []
    for r in ok:
        gold = code_set(gold_obj(gold_rows[r["index"] - 1]))
        pred_obj = parse_lenient(r["output"]) or {}
        pred = code_set(pred_obj)
        desc_hit = bool(descriptions(gold_obj(gold_rows[r["index"] - 1])) & descriptions(pred_obj))
        per_row.append({
            "exact_hit": bool(gold & pred),
            "cat_hit": bool(categories(gold) & categories(pred)),
            "desc_hit": desc_hit,
            "n_pred": len(pred),
            "chars": len(r["output"]),
            "latency_s": r["latency_s"],
        })
    n = len(per_row)
    lat = [p["latency_s"] for p in per_row]
    return {
        "model": label,
        "rows_total": n_all,
        "rows_scored": n,
        "request_failures": n_all - n,
        "strict_valid": sum(r["valid"] for r in ok),
        "lenient_valid": sum(r["lenient_valid"] for r in ok),
        "code_exact_hit_rate": round(sum(p["exact_hit"] for p in per_row) / n, 3),
        "code_category_hit_rate": round(sum(p["cat_hit"] for p in per_row) / n, 3),
        "code_description_hit_rate": round(sum(p["desc_hit"] for p in per_row) / n, 3),
        "avg_codes_predicted": round(st.mean(p["n_pred"] for p in per_row), 2),
        "latency_mean_s": round(st.mean(lat), 1),
        "latency_median_s": round(st.median(lat), 1),
        "latency_p90_s": round(sorted(lat)[int(0.9 * (n - 1))], 1),
        "latency_max_s": round(max(lat), 1),
        "median_output_chars": int(st.median(p["chars"] for p in per_row)),
        "median_s_per_1k_chars": round(st.median(p["latency_s"] / p["chars"] * 1000 for p in per_row), 1),
    }


def gold_audit(gold_rows):
    codes = [c for r in gold_rows for c in code_set(gold_obj(r))]
    return {
        "gold_codes": len(codes),
        "distinct": len(set(codes)),
        "icd10_shaped": sum(bool(ICD10_SHAPE.match(c)) for c in codes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold_rows = load_jsonl(args.test)
    out = {
        "base": score(json.load(open(args.base, encoding="utf-8"))["results"], gold_rows, "base"),
        "finetuned": score(json.load(open(args.finetuned, encoding="utf-8"))["results"], gold_rows, "finetuned"),
        "gold_audit": gold_audit(gold_rows),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    keys = [k for k in out["base"] if k != "model"]
    print(f"{'metric':28}{'base':>10}{'fine-tuned':>12}")
    for k in keys:
        print(f"{k:28}{out['base'][k]!s:>10}{out['finetuned'][k]!s:>12}")
    print("\ngold audit:", out["gold_audit"])


if __name__ == "__main__":
    main()
