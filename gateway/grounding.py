"""Flag specifics in a note that the transcript never mentions.

Ported from evals/hallucination_rules.py, plus a demographics check. Two severities:
  review - clinically meaningful; sets needs_review on the response
  info   - recorded as a warning only. Invented demographics (age, ethnicity, name)
           appear in almost every note the model writes, so flagging them for review
           would make the flag meaningless.
"""
import re
from dataclasses import dataclass

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
UNIT_RE = re.compile(
    r"\s*(?:(?:mg|mcg|g|ml|units?|iu|bpm|mmhg|mmol|mg/dl|kg|lbs?|cm|breaths|/min)\b|%|°)",
    re.IGNORECASE)  # \b only after word-character units: it can never match after % or °
MED_RE = re.compile(r"\b([A-Z][a-z]{3,})(?:\s+[a-z]+)?\s+(\d+(?:\.\d+)?)\s?(mg|mcg|g|ml|units?|iu)\b")
DR_RE = re.compile(r"\bDr\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")
AGE_RE = re.compile(r"\b(\d{1,3})[- ]year[- ]old\b", re.IGNORECASE)
TITLE_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss)\.?\s+([A-Z][a-z]+)")
ETHNICITIES = ["Caucasian", "African American", "Hispanic", "Latino", "Latina", "Asian",
               "Native American", "Pacific Islander", "Middle Eastern"]
MED_STOP = {"Blood", "Heart", "Temperature", "Weight", "Height", "Pressure", "Pain", "Oxygen", "Return"}


@dataclass(frozen=True)
class GatewayWarning:
    type: str
    severity: str  # "review" | "info"
    detail: str
    category: str = "content"  # "code" (ICD-10 problems) | "content" (note vs. transcript)


def note_text(note):
    text = "\n".join(note[k] for k in SOAP_FIELDS if isinstance(note.get(k), str))
    return re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)  # drop numbered-list markers


def transcript_numbers(transcript):
    nums = set(NUM_RE.findall(transcript))
    for w in re.findall(r"[a-z]+", transcript.lower()):
        if w in NUMBER_WORDS:
            nums.add(str(NUMBER_WORDS[w]))
    return nums


def check_grounding(transcript, note):
    """Return warnings for note content the transcript does not support."""
    text = note_text(note)
    low = transcript.lower()
    have = transcript_numbers(transcript)
    out = []

    values = sorted({m.group(0) for m in NUM_RE.finditer(text)
                     if UNIT_RE.match(text, m.end()) and m.group(0) not in have})
    if values:
        out.append(GatewayWarning("ungrounded_measured_value", "review",
                           "values not in the transcript: " + ", ".join(values[:8])))

    meds = sorted({f"{m.group(1)} {m.group(2)}" for m in MED_RE.finditer(text)
                   if m.group(1) not in MED_STOP
                   and (m.group(1).lower() not in low or m.group(2) not in have)})
    if meds:
        out.append(GatewayWarning("ungrounded_medication", "review",
                           "medication or dose not in the transcript: " + ", ".join(meds[:5])))

    people = sorted({m.group(1) for m in DR_RE.finditer(text) if m.group(1).lower() not in low})
    if people:
        out.append(GatewayWarning("ungrounded_person", "review",
                           "doctor named but not in the transcript: " + ", ".join(people[:5])))

    demo = []
    demo += [f"{m.group(1)}-year-old" for m in AGE_RE.finditer(text) if m.group(1) not in have]
    demo += [e for e in ETHNICITIES if e.lower() in text.lower() and e.lower() not in low]
    demo += [m.group(0) for m in TITLE_RE.finditer(text) if m.group(1).lower() not in low]
    if demo:
        out.append(GatewayWarning("invented_demographic", "info",
                           "patient details not in the transcript: " + ", ".join(sorted(set(demo)))))
    return out
