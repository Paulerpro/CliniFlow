"""Transcript in, checked note out: generate, validate, retry, repair codes, flag."""
from dataclasses import dataclass, field

from .config import Settings
from .grounding import GatewayWarning, check_grounding
from .icd10 import IcdTable, resolve_code
from .schema import SOAP_FIELDS, parse_reply
from .tracing import NoopTracer

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


async def generate_note(transcript, llm, table: IcdTable, system_prompt, settings: Settings, tracer=None):
    """Generate, validate, retry, repair and flag. `tracer` (gateway/tracing.py) is optional."""
    tracer = tracer or NoopTracer()
    with tracer.span("generate_note", input=transcript) as root:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": transcript}]
        obj, problems = None, []
        attempts = 0
        for attempts in range(1, settings.max_retries + 2):
            with tracer.span("llm_call", input=messages, as_type="generation", model=settings.model_name,
                             metadata={"attempt": attempts}) as call:
                reply = await llm.complete(messages)
                obj, problems = parse_reply(reply, settings.allow_lenient_parse)
                call.update(output=reply, metadata={"attempt": attempts, "format_problems": problems},
                            **({"level": "WARNING", "status_message": ", ".join(problems)} if problems else {}))
            if not problems:
                break
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": RETRY_MESSAGE.format(problems=", ".join(problems))},
            ]
        if problems:
            root.update(level="ERROR", status_message=f"no usable reply after {attempts} attempt(s)",
                        metadata={"attempts": attempts, "format_problems": problems})
            raise PipelineError(attempts, problems)

        warnings = []
        codes = []
        with tracer.span("repair_codes", input=obj["suggested_codes"]) as span:
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
            span.update(output=[c["status"] for c in codes], metadata={"statuses": [c["status"] for c in codes]})

        note = {k: obj[k] for k in SOAP_FIELDS}
        with tracer.span("grounding", input=note) as span:
            found = check_grounding(transcript, note)
            warnings += found
            span.update(output=[w.type for w in found], metadata={"warning_types": [w.type for w in found]})

        result = NoteResult(note=note, suggested_codes=codes, warnings=warnings, attempts=attempts)
        root.update(output=result.note, metadata={
            "attempts": attempts, "code_review": result.code_review, "content_review": result.content_review,
            "warning_types": [w.type for w in warnings], "code_statuses": [c["status"] for c in codes]})
        root.score("needs_review", result.needs_review)
        root.score("code_review", result.code_review)
        root.score("content_review", result.content_review)
        root.score("attempts", attempts)
        root.score("codes_repaired", sum(c["status"] == "repaired" for c in codes))
        return result


def _code_warnings(r):
    if r.status == "repaired":
        extra = " (several equally close matches; the nearest to the model's code was used)" if r.ambiguous else ""
        found = [GatewayWarning("code_repaired", "review",
                                f"model gave {r.original_code}; replaced with {r.code} from its description{extra}",
                                category="code")]
        if r.billable is False:  # the description matched a category, not a billable code
            found.append(GatewayWarning("code_not_billable", "info",
                                        f"{r.code} is a parent code and is not billable; a more specific code is needed",
                                        category="code"))
        return found
    if r.status == "verified_non_billable":
        return [GatewayWarning("code_not_billable", "info", f"{r.code} is a parent code and is not billable", category="code")]
    if r.status == "unresolved_mismatch":
        return [GatewayWarning("code_description_mismatch", "review",
                        f"{r.code} exists but does not match the description given; kept as returned", category="code")]
    if r.status == "unresolved_not_found":
        return [GatewayWarning("code_not_found", "review",
                        f"{r.code} is not an ICD-10-CM code and no close description match was found", category="code")]
    return []
