"""Validate CliniFlow train/val/test JSONL splits against the SOAP target schema.

Standard library only, so it runs in the stock SageMaker sklearn container
with no extra installs.

Local run:
    python registry/validate_dataset.py --input-dir training/colab --output-dir registry/report

SageMaker Processing run: see registry/launch_processing_job.py, which mounts
the S3 dataset prefix at /opt/ml/processing/input and writes the report to
/opt/ml/processing/output.
"""
import argparse
import json
import os
import sys
from collections import Counter

SPLITS = ["train.jsonl", "val.jsonl", "test.jsonl"]
SOAP_FIELDS = ["subjective", "objective", "assessment", "plan"]
CODE_FIELDS = ["code", "system", "description", "confidence"]


def check_record(record):
    """Return a list of problems with one record. Empty list means valid."""
    problems = []
    if not isinstance(record.get("input"), str) or not record["input"].strip():
        problems.append("input_missing_or_empty")

    output = record.get("output")
    if isinstance(output, str):
        # Some rows may store the note as a JSON string. Parse it so it's checked too.
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return problems + ["output_string_not_json"]
    if not isinstance(output, dict):
        return problems + ["output_not_object"]

    for field in SOAP_FIELDS:
        if not isinstance(output.get(field), str):
            problems.append(f"soap_{field}_not_string")

    codes = output.get("suggested_codes")
    if not isinstance(codes, list) or not codes:
        problems.append("suggested_codes_missing_or_empty")
    else:
        for c in codes:
            if not isinstance(c, dict) or any(not isinstance(c.get(f), str) for f in CODE_FIELDS):
                problems.append("suggested_code_malformed")
                break
    return problems


def validate_split(path):
    counts = Counter()
    problems = Counter()
    rows = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                problems["line_not_json"] += 1
                continue
            issues = check_record(record)
            if issues:
                counts["invalid"] += 1
                for issue in issues:
                    problems[issue] += 1
            else:
                counts["valid"] += 1
    return {
        "rows": rows,
        "valid": counts["valid"],
        "invalid": counts["invalid"],
        "problems": dict(problems),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    report = {}
    failed = False
    for split in SPLITS:
        path = os.path.join(args.input_dir, split)
        if not os.path.exists(path):
            report[split] = {"error": "file_not_found"}
            failed = True
            continue
        result = validate_split(path)
        report[split] = result
        if result["invalid"]:
            failed = True
        print(f"{split}: rows={result['rows']} valid={result['valid']} invalid={result['invalid']}")

    with open(os.path.join(args.output_dir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if failed:
        print("VALIDATION FAILED - see validation_report.json")
        sys.exit(1)
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
