"""Parse and validate the model's reply against the SOAP target schema.

Mirrors registry/validate_dataset.py and evals/run_eval.py (the gateway is
deliberately self-contained so it can ship without the other pillars).
"""
import json
import re

SOAP_FIELDS = ["subjective", "objective", "assessment", "plan"]
CODE_FIELDS = ["code", "system", "description", "confidence"]


def parse_reply(text, allow_lenient=False):
    """Return (object or None, problems). Strict: the reply must be bare JSON."""
    text = (text or "").strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = _extract_json(text) if allow_lenient else None
        if obj is None:
            return None, ["reply_not_bare_json"]
    if not isinstance(obj, dict):
        return None, ["reply_not_a_json_object"]
    return obj, schema_problems(obj)


def _extract_json(text):
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


def schema_problems(obj):
    problems = [f"{f}_not_a_string" for f in SOAP_FIELDS if not isinstance(obj.get(f), str)]
    codes = obj.get("suggested_codes")
    if not isinstance(codes, list) or not codes:
        problems.append("suggested_codes_missing_or_empty")
    elif any(not isinstance(c, dict) or any(not isinstance(c.get(f), str) for f in CODE_FIELDS) for c in codes):
        problems.append("suggested_code_malformed")
    return problems
