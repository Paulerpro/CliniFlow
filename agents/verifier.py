"""Independent check of a draft note against its transcript, by a second model.

Same rubric as evals/hallucination_judge.py (which measured it on 149 notes): list every
claim in the note the transcript does not support, tagged by type and severity, plus the
important transcript facts the note leaves out. The gateway's own content check catches
only invented numbers, drugs and doctor names; this one reads the whole note, so it also
catches things like a wrong drug-condition pairing.

It is a second opinion, not ground truth: one model, one pass, and it can be wrong in
both directions.
"""
import json

import httpx

from . import config

CLAIM_TYPES = [
    "fabricated_finding", "fabricated_medication", "fabricated_history", "fabricated_demographic",
    "fabricated_assessment", "fabricated_plan_item", "contradiction",
]

SYSTEM_PROMPT = f"""You audit AI-written clinical notes against the doctor-patient transcript they came from.

You receive a TRANSCRIPT and a NOTE (SOAP format). Judge ONLY whether the NOTE is supported by the TRANSCRIPT.

List every specific claim in the NOTE that the TRANSCRIPT does not support. Use these types:
{json.dumps(CLAIM_TYPES)}

Rules:
- A claim is supported if the transcript states it or it follows directly from what is stated. Rewording, abbreviating and reorganising are fine.
- Generic boilerplate that adds no specific fact (e.g. "follow up as needed") is NOT a claim.
- A specific value, drug, dose, test, referral, name, date or finding that never appears in the transcript IS an unsupported claim.
- If the note contradicts the transcript, report it as "contradiction".
- Severity: "major" if a clinician acting on it could be misled (wrong/invented drug, dose, vital, diagnosis-relevant finding, contradiction); "minor" otherwise.
- Do not judge writing style, formatting, or ICD codes.

Also list the clinically important facts stated in the TRANSCRIPT that the NOTE leaves out or gets wrong (symptoms, durations, vitals, medications with doses, tests ordered, follow-up). Be strict: list each missing fact separately, at most 8, and use an empty list only if nothing important is missing.

Reply with JSON only:
{{"unsupported_claims": [{{"claim": "<short quote or paraphrase from the note>", "type": "<type>", "severity": "major|minor"}}], "missing_key_facts": ["<fact>"], "rationale": "<one sentence>"}}"""


class VerifierError(Exception):
    """The verifier is not configured or its model call failed."""


def _note_text(note):
    return "\n".join(note[k] for k in ("subjective", "objective", "assessment", "plan"))


def _clean(result):
    claims = []
    for c in result.get("unsupported_claims", []):
        if isinstance(c, dict) and c.get("claim"):
            severity = "major" if c.get("severity") == "major" else "minor"
            claims.append({"claim": str(c["claim"]), "type": str(c.get("type", "other")), "severity": severity})
    missing = [str(m) for m in result.get("missing_key_facts", []) if m]
    return {"unsupported_claims": claims, "missing_key_facts": missing,
            "rationale": str(result.get("rationale", ""))}


async def verify(transcript, note, api_key=None, model=None, transport=None):
    """Return {unsupported_claims, missing_key_facts, rationale} for a note."""
    api_key = api_key or config.OPENAI_API_KEY
    model = model or config.VERIFIER_MODEL
    if not api_key:
        raise VerifierError("the verifier is off: set OPENAI_API_KEY (in the environment or .env)")
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                     {"role": "user", "content": f"TRANSCRIPT:\n{transcript}\n\nNOTE:\n{_note_text(note)}"}],
        "response_format": {"type": "json_object"},
    }
    if not model.startswith(("gpt-5", "o1", "o3", "o4")):
        body["temperature"] = 0  # reasoning-family models accept only the default
    try:
        async with httpx.AsyncClient(timeout=config.VERIFIER_TIMEOUT_S, transport=transport) as c:
            resp = await c.post("https://api.openai.com/v1/chat/completions", json=body,
                                headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
        parsed = json.loads(resp.json()["choices"][0]["message"]["content"])
        if not isinstance(parsed, dict):
            raise TypeError("the verifier reply was not a JSON object")
        return _clean(parsed)
    except httpx.HTTPStatusError as e:
        raise VerifierError(f"the verifier model returned HTTP {e.response.status_code}") from e
    except httpx.TimeoutException as e:
        raise VerifierError("the verifier model timed out") from e
    except (httpx.HTTPError, KeyError, IndexError, ValueError, TypeError) as e:
        raise VerifierError(f"the verifier failed: {type(e).__name__}") from e
