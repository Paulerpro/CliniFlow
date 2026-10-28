"""MCP server: lets an AI agent draft clinical notes and hand them to a human for sign-off.

Exactly three tools. There is deliberately NO approve tool: if the agent could approve
its own note, "human sign-off" would mean nothing. Approval is a separate command
that only a person runs (agents/cli.py).

Run (stdio, for Claude Desktop or any MCP client):
    python -m agents.mcp_server
"""
from mcp.server.fastmcp import FastMCP

from . import config, signoff
from .gateway_client import GatewayError, request_note
from .store import Store

mcp = FastMCP("cliniflow")
_store = None


def store():
    global _store
    if _store is None:
        _store = Store(config.DB_PATH)
    return _store


def _view(note):
    """What the agent sees: the note, its flags and its status. Never a way to change them."""
    reasons = [{"type": w["type"], "severity": w["severity"], "category": w["category"],
                "detail": w["detail"]} for w in note["warnings"]]
    return {
        "note_id": note["id"],
        "status": note["status"],
        "needs_human_attention": note["code_review"] or note["content_review"],
        "code_review": note["code_review"],
        "content_review": note["content_review"],
        "warnings": reasons,
        "note": note["note"],
        "reviewed_by": note["reviewed_by"],
        "review_reason": note["review_reason"],
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
    """Look up a note by id: its content, flags and current status
    (draft, awaiting_review, approved or rejected)."""
    note = store().get(note_id)
    if note is None:
        raise ValueError(f"no note with id {note_id}")
    return _view(note)


@mcp.tool()
def request_signoff(note_id: str) -> dict:
    """Send a draft to the human review queue. This does NOT approve it: only a person can,
    outside this tool. Tell the user the note is now waiting for their review."""
    try:
        return _view(signoff.request_signoff(store(), note_id))
    except signoff.SignoffError as e:
        raise ValueError(str(e)) from e


if __name__ == "__main__":
    mcp.run()
