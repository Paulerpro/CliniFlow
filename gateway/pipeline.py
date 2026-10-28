"""Transcript in, checked note out: generate, validate, retry, repair codes, flag."""
from dataclasses import dataclass, field

from .config import Settings
from .grounding import GatewayWarning, check_grounding
from .icd10 import IcdTable, resolve_code
from .schema import SOAP_FIELDS, parse_reply

RETRY_MESSAGE = (
    "Your previous reply could not be used ({problems}). "
    "Reply again with ONLY the JSON object described in the instructions: "
    "no markdown fences and no text before or after it."
)


class PipelineError(Exception):
    """The model never produced a usable reply."""

    def __init__(self, attempts, problems):
        super().__init__(f"no valid reply after {attempts} attempt(s): {', '.join(problems)}")
        self.attempts = attempts
        self.problems = problems


@dataclass
class NoteResult:
    note: dict
    suggested_codes: list[dict]
    warnings: list[GatewayWarning]
    attempts: int
    code_review: bool = field(init=False)
    content_review: bool = field(init=False)
    needs_review: bool = field(init=False)

    def __post_init__(self):
        flagged = {w.category for w in self.warnings if w.severity == "review"}
        self.code_review = "code" in flagged
        self.content_review = "content" in flagged
        self.needs_review = bool(flagged)


async def generate_note(transcript, llm, table: IcdTable, system_prompt, settings: Settings):
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": transcript}]
    obj, problems = None, []
    attempts = 0
    for attempts in range(1, settings.max_retries + 2):
        reply = await llm.complete(messages)
        obj, problems = parse_reply(reply, settings.allow_lenient_parse)
        if not problems:
            break
        messages = messages + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": RETRY_MESSAGE.format(problems=", ".join(problems))},
        ]
    if problems:
        raise PipelineError(attempts, problems)

    warnings = []
    codes = []
    for raw in obj["suggested_codes"]:
        resolved = resolve_code(table, raw["code"], raw["description"], settings.repair_threshold)
        codes.append({
            "code": resolved.code,
            "system": raw["system"],
            "description": resolved.description,
            "confidence": raw["confidence"],
            "status": resolved.status,
            "original_code": resolved.original_code if resolved.status == "repaired" else None,
        })
        warnings += _code_warnings(resolved)

    note = {k: obj[k] for k in SOAP_FIELDS}
    warnings += check_grounding(transcript, note)
    return NoteResult(note=note, suggested_codes=codes, warnings=warnings, attempts=attempts)


def _code_warnings(r):
    if r.status == "repaired":
        extra = " (several equally close matches; the nearest to the model's code was used)" if r.ambiguous else ""
        return [GatewayWarning("code_repaired", "review",
                        f"model gave {r.original_code}; replaced with {r.code} from its description{extra}", category="code")]
    if r.status == "verified_non_billable":
        return [GatewayWarning("code_not_billable", "info", f"{r.code} is a parent code and is not billable", category="code")]
    if r.status == "unresolved_mismatch":
        return [GatewayWarning("code_description_mismatch", "review",
                        f"{r.code} exists but does not match the description given; kept as returned", category="code")]
    if r.status == "unresolved_not_found":
        return [GatewayWarning("code_not_found", "review",
                        f"{r.code} is not an ICD-10-CM code and no close description match was found", category="code")]
    return []
