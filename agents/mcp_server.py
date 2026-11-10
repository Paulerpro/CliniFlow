"""MCP server: lets an AI agent draft clinical notes, check them, and hand them to a human for sign-off.

Five tools. There is deliberately NO approve tool: if the agent could approve its own note,
"human sign-off" would mean nothing. Approval is a separate command that only a person runs
(agents/cli.py).

Run (stdio, for Claude Desktop or any MCP client):
    python -m agents.mcp_server
"""
import asyncio

from mcp.server.fastmcp import FastMCP

from gateway.icd10 import IcdTable
from gateway.tracing import make_tracer

from . import config, signoff, verifier
from .gateway_client import GatewayError, request_note
from .store import Store

mcp = FastMCP("cliniflow")
_store = None
_table = None
_tracer = None


def store():
    global _store
    if _store is None:
        _store = Store(config.DB_PATH)
    return _store


def table():
    global _table
    if _table is None:
        _table = IcdTable.load()
    return _table


def tracer():
    """Langfuse tracing, a no-op unless LANGFUSE keys are set (see gateway/tracing.py)."""
    global _tracer
    if _tracer is None:
        _tracer = make_tracer()
    return _tracer


def _view(note):
    """What the agent sees: the note, its flags and its status. Never a way to change them."""
    warnings = [{"type": w["type"], "severity": w["severity"], "category": w["category"],
                 "detail": w["detail"]} for w in note["warnings"]]
    v = note["verification"]
    return {
        "note_id": note["id"],
        "status": note["status"],
        "needs_human_attention": signoff.needs_attention(note),
        "code_review": note["code_review"],
        "content_review": note["content_review"],
        "warnings": warnings,
        "verification": None if v is None else {
            "model": v["model"], "unsupported_claims": v["unsupported_claims"],
            "missing_key_facts": v["missing_key_facts"], "rationale": v["rationale"]},
        "note": note["note"],
        "reviewed_by": note["reviewed_by"],
        "review_reason": note["review_reason"],
        "final_code": note["final_code"],
    }


@mcp.tool()
async def draft_note(transcript: str) -> dict:
    """Turn a doctor-patient transcript into a DRAFT SOAP note with suggested ICD-10 codes.

    The note is not approved and must not be treated as final: a human has to sign it off.
    Check `needs_human_attention` and `warnings` and tell the user about anything flagged.
    A cold model container can make the first call take several minutes.
    """
    try:
        response = await request_note(transcript)
    except GatewayError as e:
        raise ValueError(str(e)) from e
    note_id = store().create_draft(transcript, response)
    return _view(store().get(note_id))


@mcp.tool()
def get_note(note_id: str) -> dict:
    """Look up a note by id: its content, flags, independent-check results and current status
    (draft, awaiting_review, approved or rejected)."""
    note = store().get(note_id)
    if note is None:
        raise ValueError(f"no note with id {note_id}")
    return _view(note)


@mcp.tool()
async def verify_note(note_id: str) -> dict:
    """Run an independent second check of a note against its transcript, by a different model.

    It lists claims in the note the transcript does not support (major or minor) and the
    important transcript facts the note leaves out. A major finding means a reviewer must give a
    written reason to approve the note. This is a second opinion, not proof: report its findings
    to the user and do not "fix" the note yourself. Needs OPENAI_API_KEY; sends the transcript and
    note to OpenAI, so use it with synthetic data only.
    """
    note = store().get(note_id)
    if note is None:
        raise ValueError(f"no note with id {note_id}")
    if note["status"] not in ("draft", "awaiting_review"):
        raise ValueError(f"note is {note['status']}; it can no longer be verified")
    try:
        with tracer().span("verify_note", input=note["transcript"], as_type="generation",
                           model=config.VERIFIER_MODEL, metadata={"note_id": note_id}) as span:
            result = await verifier.verify(note["transcript"], note["note"])
            major = sum(c["severity"] == "major" for c in result["unsupported_claims"])
            span.update(output=result, metadata={"note_id": note_id, "major_claims": major})
            span.score("verifier_major_claims", major)
            span.score("verifier_missing_facts", len(result["missing_key_facts"]))
    except verifier.VerifierError as e:
        raise ValueError(str(e)) from e
    # Send the trace now: the background uploader runs every few seconds and a short-lived
    # server (a client that exits right after the call) would otherwise lose it.
    await asyncio.to_thread(tracer().flush)
    try:
        return _view(signoff.record_verification(store(), note_id, result, config.VERIFIER_MODEL))
    except signoff.SignoffError as e:
        raise ValueError(str(e)) from e


@mcp.tool()
def search_icd10(query: str, limit: int = 5) -> list[dict]:
    """Search the official ICD-10-CM code list (FY2026) by diagnosis wording, e.g.
    "type 2 diabetes with diabetic chronic kidney disease". Returns the best-matching billable codes
    with a 0-100 match score. Works best with clinical diagnosis phrases; plain symptom words rank
    poorly. Read-only: use it to suggest candidates to the reviewer, not to change a note."""
    limit = max(1, min(int(limit), 20))
    return [{"code": e.code, "description": e.description, "match_score": score}
            for e, score in table().search(query, limit=limit)]


@mcp.tool()
def request_signoff(note_id: str) -> dict:
    """Send a draft to the human review queue. This does NOT approve it: only a person can,
    outside this tool. Tell the user the note is now waiting for their review."""
    try:
        return _view(signoff.request_signoff(store(), note_id))
    except signoff.SignoffError as e:
        raise ValueError(str(e)) from e


if __name__ == "__main__":
    tracer()  # start tracing before serving: lazy SDK imports stall inside the stdio server
    try:
        mcp.run()
    finally:
        tracer().flush()  # the client closed stdin: send whatever is still buffered
