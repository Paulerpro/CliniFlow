"""Print a side-by-side comparison of two run_eval.py result files.

Usage:
    python evals/compare.py evals/results_base.json evals/results_finetuned.json
"""
import json
import sys
from collections import Counter

METRICS = [
    ("valid", "strict valid (bare JSON + schema)"),
    ("lenient_valid", "lenient valid (fences stripped)"),
    ("code_hit_rate", "gold code hit rate"),
    ("code_exact_rate", "gold code exact-set rate"),
    ("code_precision", "code precision"),
    ("code_recall", "code recall"),
    ("code_f1", "code F1"),
    ("avg_latency_s", "avg latency (s)"),
]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def top_problems(data, key, n=5):
    counts = Counter(p for r in data["results"] for p in r[key])
    return counts.most_common(n)


def main():
    base_path, ft_path = sys.argv[1], sys.argv[2]
    base, ft = load(base_path), load(ft_path)
    print(f"{'metric':38}{'base':>10}{'fine-tuned':>12}")
    for key, label in METRICS:
        print(f"{label:38}{base['summary'][key]!s:>10}{ft['summary'][key]!s:>12}")
    for name, data in (("base", base), ("fine-tuned", ft)):
        print(f"\n{name} top strict failures:  {top_problems(data, 'problems')}")
        print(f"{name} top lenient failures: {top_problems(data, 'lenient_problems')}")


if __name__ == "__main__":
    main()
