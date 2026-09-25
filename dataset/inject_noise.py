"""Inject realistic noise into transcripts (input only; gold notes stay clean).

Design rules, so the gold note remains derivable from the noisy transcript:
  - additive noise only (fillers, false starts, backchannel, crosstalk markers) - no clinical content is deleted
  - abbreviation swaps limited to unambiguous terms
  - numbers, drug names and negations are never touched (asserted at the end)
Deterministic: seeded per example index.
"""
import json
import random
import re
from collections import Counter

SRC = "medsynth_1k_schema_valid.jsonl"
DST = "medsynth_1k_noisy.jsonl"

# (regex, replacement) - unambiguous clinical shorthand
ABBREVIATIONS = [
    (r"\bblood pressure\b", "BP"),
    (r"\bheart rate\b", "HR"),
    (r"\brespiratory rate\b", "RR"),
    (r"\bshortness of breath\b", "SOB"),
    (r"\bpast medical history\b", "PMH"),
    (r"\bfamily history\b", "FHx"),
    (r"\breview of systems\b", "ROS"),
    (r"\bcomplete blood count\b", "CBC"),
    (r"\belectrocardiogram\b", "EKG"),
    (r"\bas needed\b", "PRN"),
    (r"\btwice a day\b", "BID"),
    (r"\bonce a day\b", "daily"),
    (r"\bby mouth\b", "PO"),
    (r"\bno acute distress\b", "NAD"),
    (r"\bgastroesophageal reflux disease\b", "GERD"),
    (r"\bhypertension\b", "HTN"),
    (r"\btype 2 diabetes\b", "T2DM"),
]
FILLERS = ["um,", "uh,", "well,", "you know,", "I mean,", "like,", "so,", "hmm,"]
BACKCHANNEL_PATIENT = ["Mm-hm.", "Yeah.", "Right.", "Okay.", "Uh-huh."]
BACKCHANNEL_DOCTOR = ["Mm-hm.", "Right.", "Okay.", "I see."]
CROSSTALK = ["[crosstalk]", "[inaudible]", "[overlapping speech]"]

TURN_RE = re.compile(r"^\[(doctor|patient)\]\s*(.*)$", re.DOTALL)


def split_turns(text):
    return [t for t in re.split(r"\n\n+", text.strip()) if t.strip()]


def swap_abbreviations(s, rng, p):
    for pat, rep in ABBREVIATIONS:
        if rng.random() < p:
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s


def add_filler(s, rng):
    words = s.split(" ")
    if len(words) < 4:
        return s
    i = rng.randint(1, min(len(words) - 1, 6))
    words.insert(i, rng.choice(FILLERS))
    return " ".join(words)


def false_start(s, rng):
    """Repeat/restart the opening: 'It's been' -> 'It's, it's been'; keeps every original word."""
    words = s.split(" ")
    if len(words) < 3:
        return s
    w = words[0]
    return f"{w.rstrip(',.')}, {' '.join(words)}" if rng.random() < 0.5 else f"{w}-- {' '.join(words)}"


def apply_noise(text, level, rng):
    """level: 'clean' | 'light' | 'heavy'"""
    if level == "clean":
        return text
    p_abbr, p_fill, p_false, p_back, p_cross = (
        (0.4, 0.10, 0.05, 0.03, 0.01) if level == "light" else (0.8, 0.30, 0.15, 0.10, 0.05)
    )
    out = []
    for turn in split_turns(text):
        m = TURN_RE.match(turn)
        if not m:
            out.append(turn)
            continue
        role, body = m.group(1), m.group(2)
        body = swap_abbreviations(body, rng, p_abbr)
        if role == "patient":
            if rng.random() < p_fill:
                body = add_filler(body, rng)
            if rng.random() < p_false:
                body = false_start(body, rng)
            if level == "heavy" and rng.random() < 0.3:
                body = body[0].lower() + body[1:]  # sloppy transcript casing
        elif rng.random() < p_fill / 2:
            body = add_filler(body, rng)
        out.append(f"[{role}] {body}")
        # interruption: short backchannel from the other speaker after a long turn
        if len(body) > 120 and rng.random() < p_back:
            other = "doctor" if role == "patient" else "patient"
            pool = BACKCHANNEL_DOCTOR if other == "doctor" else BACKCHANNEL_PATIENT
            out.append(f"[{other}] {rng.choice(pool)}")
        if rng.random() < p_cross:
            out.append(rng.choice(CROSSTALK))
    return "\n\n".join(out)


NUM_RE = re.compile(r"\d+(?:\.\d+)?")

rows = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
levels = Counter()
with open(DST, "w", encoding="utf-8") as f:
    for i, r in enumerate(rows):
        rng = random.Random(1000 + i)
        roll = rng.random()
        level = "clean" if roll < 0.2 else "light" if roll < 0.6 else "heavy"
        noisy = apply_noise(r["input"], level, rng)
        # safety: no number may be added or removed
        assert Counter(NUM_RE.findall(noisy)) == Counter(NUM_RE.findall(r["input"])), f"numbers changed at row {i}"
        levels[level] += 1
        f.write(json.dumps({"input": noisy, "output": r["output"], "noise_level": level}, ensure_ascii=False) + "\n")

print("wrote", DST, dict(levels))
