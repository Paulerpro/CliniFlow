"""Measure what the gateway adds: replay the saved model outputs through the real
gateway pipeline (no model server, no API calls) and score the result.

Reports, before vs. after the gateway: exact / category / description code hits, and
how many notes get flagged for code review and for content review. If the LLM-judge
results exist, it also checks how many of the notes the judge found a major
unsupported claim in the content check catches, and how many it flags otherwise.

Usage:
    python evals/gateway_replay.py --test training/colab/test.jsonl \
        --results evals/results_finetuned.json --judge evals/hallucination_judge.json \
        --out evals/gateway_replay.json
"""
import argparse
import asyncio
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from gateway.config import Settings  # noqa: E402
from gateway.icd10 import IcdTable, normalize_code  # noqa: E402
from gateway.pipeline import PipelineError, generate_note  # noqa: E402


class ReplayLLM:
    def __init__(self, reply):
        self.reply = reply

    async def complete(self, messages):
        return self.reply


def category(code):
    return normalize_code(code)[:3]


def hits(pred_codes, gold_code, gold_desc):
    pred = [normalize_code(c["code"]) for c in pred_codes]
    gold = normalize_code(gold_code)
    return {
        "exact": gold in pred,
        "category": any(p[:3] == gold[:3] for p in pred),
        "description": any(c["description"].upper() == gold_desc.upper() for c in pred_codes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--judge", default=None)
    ap.add_argument("--judge-label", default="finetuned", help="which source in the judge file holds these results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.test, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(args.results, encoding="utf-8") as f:
        results = json.load(f)["results"]
    table = IcdTable.load()
    settings = Settings(max_retries=0)  # replay has one saved reply per row; nothing to retry

    before, after = Counter(), Counter()
    flagged, code_flagged, content_flagged = [], [], []
    warn_types, failed = Counter(), []
    scored = 0
    for res, row in zip(results, rows):
        if res["output"] is None:
            continue
        gold = row["output"]["suggested_codes"][0]
        try:
            out = asyncio.run(generate_note(row["input"], ReplayLLM(res["output"]), table, "", settings))
        except PipelineError as e:
            failed.append((res["index"], e.problems))
            continue
        scored += 1
        raw = json.loads(res["output"])["suggested_codes"]
        for k, v in hits(raw, gold["code"], gold["description"]).items():
            before[k] += v
        for k, v in hits(out.suggested_codes, gold["code"], gold["description"]).items():
            after[k] += v
        for w in out.warnings:
            warn_types[f"{w.type} ({w.severity})"] += 1
        if out.needs_review:
            flagged.append(res["index"])
        if out.code_review:
            code_flagged.append(res["index"])
        if out.content_review:
            content_flagged.append(res["index"])

    summary = {
        "notes_scored": scored,
        "unusable_replies": failed,
        "code_hits_before_gateway": dict(before),
        "code_hits_after_gateway": dict(after),
        "needs_review": len(flagged),
        "code_review": len(code_flagged),
        "content_review": len(content_flagged),
        "warning_counts": dict(warn_types),
    }

    if args.judge:
        with open(args.judge, encoding="utf-8") as f:
            judgments = json.load(f)["judgments"]
        label = args.judge_label
        major = {int(k.split(":")[1]) for k, v in judgments.items() if k.startswith(label + ":")
                 and any(c["severity"] == "major" for c in v["unsupported_claims"])}
        summary["judge_major_notes"] = len(major)
        summary["content_review_on_judge_major"] = len(major & set(content_flagged))
        summary["content_review_not_judge_major"] = len(set(content_flagged) - major)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
