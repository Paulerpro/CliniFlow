"""Rule-based hallucination checks: does each SOAP note contain specifics that
the transcript never mentions?

Three checks, all comparing the note against its source transcript:
  - numbers:    every number in the note should appear in the transcript
                (spelled-out numbers in the transcript are normalised)
  - medications: "<Drug> <dose><unit>" in the note; drug name and dose must
                both appear in the transcript
  - people:     "Dr. <Name>" in the note must appear in the transcript

The gold notes in the test set are themselves elaborated beyond the transcript
(added exam findings, tablet counts, etc.), so the gold outputs are scored as a
control: a model is judged against that baseline, not against zero.

Usage:
    python evals/hallucination_rules.py --test training/colab/test.jsonl \
        --base evals/results_base.json --finetuned evals/results_finetuned.json \
        --out evals/hallucination_rules.json
"""
import argparse
import json
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import parse_lenient  # noqa: E402

SOAP_FIELDS = ["subjective", "objective", "assessment", "plan"]

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100,
}
NUM_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
UNIT_RE = re.compile(r"\s*(mg|mcg|g|ml|units?|iu|bpm|mmhg|mmol|mg/dl|%|°|kg|lbs?|cm|breaths|/min)\b",
                     re.IGNORECASE)
MED_RE = re.compile(r"\b([A-Z][a-z]{3,})(?:\s+[a-z]+)?\s+(\d+(?:\.\d+)?)\s?(mg|mcg|g|ml|units?|iu)\b")
DR_RE = re.compile(r"\bDr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
# words that look like drug names to MED_RE but are not
MED_STOP = {"Blood", "Heart", "Temperature", "Weight", "Height", "Pressure", "Pain", "Oxygen", "Return"}


def note_text(note):
    parts = [note.get(k, "") for k in SOAP_FIELDS if isinstance(note.get(k), str)]
    text = "\n".join(parts)
    return re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)  # drop numbered-list markers


def transcript_numbers(transcript):
    nums = set(NUM_RE.findall(transcript))
    for w in re.findall(r"[a-z]+", transcript.lower()):
        if w in NUMBER_WORDS:
            nums.add(str(NUMBER_WORDS[w]))
    return nums


def number_check(transcript, text):
    have = transcript_numbers(transcript)
    found = []  # (number, has_unit)
    for m in NUM_RE.finditer(text):
        found.append((m.group(0), bool(UNIT_RE.match(text, m.end()))))
    ungrounded = [(n, u) for n, u in found if n not in have]
    measured = [(n, u) for n, u in found if u]
    return {
        "total": len(found),
        "ungrounded": len(ungrounded),
        "measured_total": len(measured),
        "measured_ungrounded": sum(1 for n, u in ungrounded if u),
        "examples": sorted({n for n, _ in ungrounded})[:8],
    }


def medication_check(transcript, text):
    low = transcript.lower()
    meds, bad = set(), []
    for m in MED_RE.finditer(text):
        drug, dose = m.group(1), m.group(2)
        if drug in MED_STOP:
            continue
        meds.add((drug, dose))
        if drug.lower() not in low or dose not in transcript_numbers(transcript):
            bad.append(f"{drug} {dose}")
    return {"total": len(meds), "ungrounded": len(set(bad)), "examples": sorted(set(bad))[:5]}


def people_check(transcript, text):
    low = transcript.lower()
    names = {m.group(1) for m in DR_RE.finditer(text)}
    bad = [n for n in names if n.lower() not in low]
    return {"total": len(names), "ungrounded": len(bad), "examples": sorted(bad)[:5]}


def check_note(transcript, note):
    text = note_text(note)
    return {
        "numbers": number_check(transcript, text),
        "medications": medication_check(transcript, text),
        "people": people_check(transcript, text),
    }


def summarise(per_note, label):
    n = len(per_note)
    num_rates = [c["numbers"]["ungrounded"] / c["numbers"]["total"]
                 for c in per_note if c["numbers"]["total"]]
    return {
        "model": label,
        "notes_scored": n,
        "mean_ungrounded_number_rate": round(st.mean(num_rates), 3) if num_rates else None,
        "mean_ungrounded_numbers_per_note": round(st.mean(c["numbers"]["ungrounded"] for c in per_note), 2),
        "mean_ungrounded_measured_values_per_note":
            round(st.mean(c["numbers"]["measured_ungrounded"] for c in per_note), 2),
        "notes_with_ungrounded_medication":
            sum(1 for c in per_note if c["medications"]["ungrounded"]),
        "notes_with_ungrounded_person": sum(1 for c in per_note if c["people"]["ungrounded"]),
    }


def load_results(path, rows):
    out = {}
    for r in json.load(open(path, encoding="utf-8"))["results"]:
        if r["output"] is None:
            continue
        obj = parse_lenient(r["output"])
        if obj is not None:
            out[r["index"] - 1] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.test, encoding="utf-8") if l.strip()]
    sources = {
        "gold": {i: (json.loads(r["output"]) if isinstance(r["output"], str) else r["output"])
                 for i, r in enumerate(rows)},
        "base": load_results(args.base, rows),
        "finetuned": load_results(args.finetuned, rows),
    }
    out = {"summary": {}, "per_note": {}}
    for label, notes in sources.items():
        per_note = {i: check_note(rows[i]["input"], note) for i, note in notes.items()}
        out["per_note"][label] = {str(i + 1): c for i, c in per_note.items()}
        out["summary"][label] = summarise(list(per_note.values()), label)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    keys = [k for k in out["summary"]["gold"] if k != "model"]
    print(f"{'metric':44}" + "".join(f"{m:>11}" for m in sources))
    for k in keys:
        print(f"{k:44}" + "".join(f"{out['summary'][m][k]!s:>11}" for m in sources))


if __name__ == "__main__":
    main()
