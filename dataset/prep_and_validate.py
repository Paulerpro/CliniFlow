import json
import re
import unicodedata
from collections import Counter
from pydantic import BaseModel

# 1. Define Target Schema
class SuggestedCode(BaseModel):
    code: str
    system: str
    description: str
    confidence: str

class TargetSoapSchema(BaseModel):
    subjective: str
    objective: str
    assessment: str
    plan: str
    suggested_codes: list[SuggestedCode]

# 2. Text Normalization
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()

# 3. Flexible heading pattern builder
# Handles all observed variants:
#   "1. **Subjective:**"   (number outside bold)
#   "**1. Subjective:**"   (number inside bold)
#   "**Subjective:**"      (no number)
# Tolerates optional colon, optional bold markers, optional leading number
# either outside or inside the ** markers.
def heading_pattern(name: str) -> str:
    return (
        r"(?:\d+\.\s*)?"      # optional "1. " before the **
        r"\*{0,2}\s*"         # optional ** (0, 1, or 2 asterisks, some sources use single)
        r"(?:\d+\.\s*)?"      # optional "1. " inside the **
        rf"{name}\s*:?\s*"    # the section name itself, optional colon
        r"\*{0,2}"            # closing ** (optional)
    )

SECTION_NAMES = ["Subjective", "Objective", "Assessment", "Plan"]

def parse_markdown_soap(note_text: str) -> dict:
    cleaned = clean_text(note_text)

    boundaries = [heading_pattern(n) for n in SECTION_NAMES]
    parsed = {}
    for i, name in enumerate(SECTION_NAMES):
        start_pat = boundaries[i]
        # lookahead = any of the remaining section headings, or end of string
        remaining = boundaries[i + 1:]
        lookahead = "|".join(remaining) if remaining else None
        if lookahead:
            pattern = rf"{start_pat}(.*?)(?={lookahead}|$)"
        else:
            pattern = rf"{start_pat}(.*?)$"
        match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
        parsed[name.lower()] = match.group(1).strip() if match else ""

    return parsed

# 4. Main Conversion Loop
INPUT_FILE = "medsynth_1k_soap.jsonl"
OUTPUT_CLEAN_FILE = "medsynth_1k_schema_valid.jsonl"
OUTPUT_DROPPED_FILE = "medsynth_1k_dropped.jsonl"

# One source row (line 309, a COPD note) has an empty "Objective" section, so validation rejects it and
# it is still listed in the dropped file. The dataset used for training and evaluation kept that row,
# with the objective written as below, appended as the last row. The repair is recorded here so the data
# can be regenerated exactly: without it the row count, the shuffle and the whole test split change.
MANUAL_REPAIRS = {
    309: {
        "sections": {"objective": "Not documented — encounter conducted via telemedicine; no in-person vitals or physical exam performed."},
        "strip_trailing_spaces": ["subjective"],
        # MedSynth's own label for this row is an HPV-screening code, which does not fit a COPD note.
        "confidence": "low",
    },
}


def build_sample(row: dict, repair: dict | None = None) -> dict:
    """Convert one raw MedSynth row into an {input, output} pair; raises ValueError if it is unusable."""
    cleaned_dialogue = clean_text(row.get("Dialogue", ""))
    if not cleaned_dialogue:
        raise ValueError("EMPTY_DIALOGUE")

    raw_note = row.get(" Note", row.get("Note", ""))  # source has leading space in key
    parsed_soap = parse_markdown_soap(raw_note)
    repair = repair or {}
    parsed_soap.update(repair.get("sections", {}))
    for name in repair.get("strip_trailing_spaces", []):
        parsed_soap[name] = re.sub(r" +\n", "\n", parsed_soap[name])

    missing_sections = [k for k, v in parsed_soap.items() if not v]
    if missing_sections:
        raise ValueError(f"MISSING_SECTIONS:{','.join(missing_sections)}")

    icd_code = str(row.get("ICD10", "")).strip()
    icd_desc = str(row.get("ICD10_desc", "")).strip()

    suggested_codes = []
    if icd_code:
        suggested_codes.append({
            "code": icd_code,
            "system": "ICD-10",
            "description": icd_desc,
            "confidence": repair.get("confidence", "high")
        })

    formatted_target = {
        "subjective": parsed_soap["subjective"],
        "objective": parsed_soap["objective"],
        "assessment": parsed_soap["assessment"],
        "plan": parsed_soap["plan"],
        "suggested_codes": suggested_codes
    }

    validated_soap = TargetSoapSchema(**formatted_target)
    return {"input": cleaned_dialogue, "output": validated_soap.model_dump()}


valid_count = 0
dropped_count = 0
drop_reasons = Counter()

with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_CLEAN_FILE, "w", encoding="utf-8") as outfile, \
     open(OUTPUT_DROPPED_FILE, "w", encoding="utf-8") as dropfile:

    for line_idx, line in enumerate(infile, 1):
        if not line.strip():
            continue

        try:
            final_sample = build_sample(json.loads(line))
            outfile.write(json.dumps(final_sample, ensure_ascii=False) + "\n")
            valid_count += 1

        except Exception as e:
            dropped_count += 1
            reason = str(e).split(":")[0]
            drop_reasons[reason] += 1
            error_record = {
                "line_number": line_idx,
                "error": str(e),
                "raw_data_snippet": line[:200]
            }
            dropfile.write(json.dumps(error_record, ensure_ascii=False) + "\n")

# Hand-repaired rows go last, in line order (see MANUAL_REPAIRS).
if MANUAL_REPAIRS:
    with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
         open(OUTPUT_CLEAN_FILE, "a", encoding="utf-8") as outfile:
        source_lines = {i: text for i, text in enumerate(infile, 1) if i in MANUAL_REPAIRS}
        for line_no, repair in sorted(MANUAL_REPAIRS.items()):
            outfile.write(json.dumps(build_sample(json.loads(source_lines[line_no]), repair),
                                     ensure_ascii=False) + "\n")
            valid_count += 1
            print(f"Applied manual repair to source line {line_no}")

print("--- CONVERSION COMPLETE ---")
print(f"Valid & Converted: {valid_count}")
print(f"Dropped/Flagged:   {dropped_count}")
print("\n--- DROP REASON BREAKDOWN ---")
for reason, count in drop_reasons.most_common():
    print(f"  {reason}: {count}")