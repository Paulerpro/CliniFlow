"""Compare the merged model with the adapter-served model on the same 50 rows.

Two questions:
  1. Does merging change what the model says? (output agreement at temperature 0)
  2. Is it faster? (latency, plus latency per 1,000 output characters, which removes
     differences in how much each model writes)

Usage:
    python evals/compare_merged.py --adapter evals/results_finetuned.json \
        --merged evals/results_merged.json --out evals/merged_comparison.json
"""
import argparse
import difflib
import json
import statistics as st


def load(path):
    with open(path, encoding="utf-8") as f:
        return {r["index"]: r for r in json.load(f)["results"]}


def parse(text):
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def latency_stats(rows):
    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None and r["output"]]
    per_1k = [r["latency_s"] / len(r["output"]) * 1000 for r in rows if r["latency_s"] is not None and r["output"]]
    return {
        "n": len(lat), "median_s": round(st.median(lat), 1), "mean_s": round(st.mean(lat), 1),
        "p90_s": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 1), "max_s": round(max(lat), 1),
        "median_s_per_1k_chars": round(st.median(per_1k), 1),
        "median_output_chars": int(st.median(len(r["output"]) for r in rows if r["output"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--merged", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    adapter, merged = load(args.adapter), load(args.merged)
    both = [i for i in adapter if i in merged and adapter[i]["output"] and merged[i]["output"]]

    identical, same_code, same_fields = 0, 0, 0
    similarities, diverged = [], []
    for i in both:
        a, m = adapter[i]["output"], merged[i]["output"]
        pa, pm = parse(a), parse(m)
        if a == m:
            identical += 1
        else:
            diverged.append((i, common_prefix(a, m), len(a)))
        similarities.append(difflib.SequenceMatcher(None, a, m).ratio())
        if pa and pm:
            same_code += [c["code"] for c in pa["suggested_codes"]] == [c["code"] for c in pm["suggested_codes"]]
            same_fields += all(pa[k] == pm[k] for k in ("subjective", "objective", "assessment", "plan"))

    n = len(both)
    summary = {
        "rows_compared": n,
        "outputs_byte_identical": identical,
        "soap_text_identical": same_fields,
        "suggested_codes_identical": same_code,
        "mean_text_similarity": round(st.mean(similarities), 4),
        "first_divergence_examples": [
            {"row": i, "diverges_at_char": at, "of": total} for i, at, total in diverged[:8]],
        "adapter_latency": latency_stats([adapter[i] for i in both]),
        "merged_latency": latency_stats([merged[i] for i in both]),
    }
    a_med, m_med = summary["adapter_latency"]["median_s"], summary["merged_latency"]["median_s"]
    a_k, m_k = summary["adapter_latency"]["median_s_per_1k_chars"], summary["merged_latency"]["median_s_per_1k_chars"]
    summary["speedup_median_latency"] = round(a_med / m_med, 2)
    summary["speedup_per_1k_chars"] = round(a_k / m_k, 2)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
