"""Sign-off rules. The only code that changes a note's status.

    draft --request_signoff (agent)--> awaiting_review --approve / reject (human)--> approved / rejected
    draft --reject (human)--> rejected

An agent can reach `awaiting_review` and no further. Approval is a function here, but
it is not exposed as an MCP tool; people run it through agents/cli.py.
"""
from .config import MIN_REASON_CHARS


class SignoffError(Exception):
    """A rule was broken; the message says which."""


def _get(store, note_id):
    note = store.get(note_id)
    if note is None:
        raise SignoffError(f"no note with id {note_id}")
    return note


def verifier_major(note):
    v = note.get("verification")
    return bool(v) and any(c["severity"] == "major" for c in v["unsupported_claims"])


def flags(note):
    return {"code_review": note["code_review"], "content_review": note["content_review"],
            "verifier_major": verifier_major(note)}


def needs_attention(note):
    return any(flags(note).values())


def request_signoff(store, note_id):
    """Agent action: hand a draft to the human review queue. Cannot approve anything."""
    note = _get(store, note_id)
    if note["status"] != "draft":
        raise SignoffError(f"note is {note['status']}; only a draft can be sent for sign-off")
    store.set_status(note_id, "awaiting_review", "agent", "requested_signoff", flags(note))
    return store.get(note_id)


def record_verification(store, note_id, result, model):
    """Agent action: attach an independent check. Only for notes still open to review."""
    note = _get(store, note_id)
    if note["status"] not in ("draft", "awaiting_review"):
        raise SignoffError(f"note is {note['status']}; it can no longer be verified")
    store.add_verification(note_id, model, result)
    return store.get(note_id)


def approve(store, note_id, reviewer, reason=None, final_code=None, table=None):
    note = _get(store, note_id)
    if note["status"] != "awaiting_review":
        raise SignoffError(f"note is {note['status']}; only a note awaiting review can be approved")
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise SignoffError("a reviewer name is required")
    reason = (reason or "").strip()
    raised = [k for k, v in flags(note).items() if v]
    if raised and len(reason) < MIN_REASON_CHARS:
        raise SignoffError(f"this note was flagged ({', '.join(raised)}); approving it needs a written "
                           f"reason of at least {MIN_REASON_CHARS} characters")
    code = None
    if final_code:
        if table is None:
            raise SignoffError("the ICD-10 table is needed to set a final code")
        entry = table.lookup(final_code)
        if entry is None:
            raise SignoffError(f"{final_code} is not an ICD-10-CM code")
        code = entry.code
    store.set_status(note_id, "approved", f"human:{reviewer}", "approved",
                     {**flags(note), "reason": reason, "final_code": code},
                     reviewer=reviewer, reason=reason, final_code=code)
    return store.get(note_id)


def reject(store, note_id, reviewer, reason):
    note = _get(store, note_id)
    if note["status"] not in ("draft", "awaiting_review"):
        raise SignoffError(f"note is {note['status']}; it can no longer be rejected")
    reviewer = (reviewer or "").strip()
    reason = (reason or "").strip()
    if not reviewer:
        raise SignoffError("a reviewer name is required")
    if len(reason) < MIN_REASON_CHARS:
        raise SignoffError(f"rejecting needs a written reason of at least {MIN_REASON_CHARS} characters")
    store.set_status(note_id, "rejected", f"human:{reviewer}", "rejected",
                     {**flags(note), "reason": reason}, reviewer=reviewer, reason=reason)
    return store.get(note_id)
