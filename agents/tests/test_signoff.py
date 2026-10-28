import pytest

from agents import signoff
from agents.signoff import SignoffError

from .conftest import FLAG_WARNING, gateway_response


def draft(store, flagged=False):
    gr = gateway_response(code_review=flagged, warnings=[FLAG_WARNING] if flagged else [])
    return store.create_draft("t", gr)


def test_agent_can_request_signoff_but_note_is_not_approved(store):
    nid = draft(store)
    note = signoff.request_signoff(store, nid)
    assert note["status"] == "awaiting_review" and note["reviewed_by"] is None


def test_cannot_request_signoff_twice(store):
    nid = draft(store)
    signoff.request_signoff(store, nid)
    with pytest.raises(SignoffError, match="only a draft"):
        signoff.request_signoff(store, nid)


def test_draft_cannot_be_approved_without_going_through_review(store):
    nid = draft(store)
    with pytest.raises(SignoffError, match="awaiting review"):
        signoff.approve(store, nid, "Dr. Lee", "looks right to me")


def test_clean_note_can_be_approved_without_a_reason(store):
    nid = draft(store)
    signoff.request_signoff(store, nid)
    note = signoff.approve(store, nid, "Dr. Lee")
    assert note["status"] == "approved" and note["reviewed_by"] == "Dr. Lee"


def test_flagged_note_needs_a_written_reason(store):
    nid = draft(store, flagged=True)
    signoff.request_signoff(store, nid)
    for reason in (None, "", "ok", "         "):
        with pytest.raises(SignoffError, match="written reason"):
            signoff.approve(store, nid, "Dr. Lee", reason)
    note = signoff.approve(store, nid, "Dr. Lee", "Checked the code against the chart; N809 is right.")
    assert note["status"] == "approved"


def test_reviewer_name_is_required(store):
    nid = draft(store)
    signoff.request_signoff(store, nid)
    with pytest.raises(SignoffError, match="reviewer"):
        signoff.approve(store, nid, "  ")


def test_reject_needs_a_reason_and_works_from_draft_or_queue(store):
    a, b = draft(store), draft(store)
    signoff.request_signoff(store, b)
    for nid in (a, b):
        with pytest.raises(SignoffError, match="written reason"):
            signoff.reject(store, nid, "Dr. Lee", "no")
        assert signoff.reject(store, nid, "Dr. Lee", "Invented a medication.")["status"] == "rejected"


def test_finished_notes_cannot_be_reapproved_or_rejected(store):
    nid = draft(store)
    signoff.request_signoff(store, nid)
    signoff.approve(store, nid, "Dr. Lee")
    with pytest.raises(SignoffError):
        signoff.approve(store, nid, "Dr. Lee")
    with pytest.raises(SignoffError, match="no longer be rejected"):
        signoff.reject(store, nid, "Dr. Lee", "changed my mind later")


def test_unknown_note(store):
    with pytest.raises(SignoffError, match="no note"):
        signoff.request_signoff(store, "missing")


def test_full_history_is_audited_with_who_did_what(store):
    nid = draft(store, flagged=True)
    signoff.request_signoff(store, nid)
    signoff.approve(store, nid, "Dr. Lee", "Verified against the chart.")
    rows = store.audit(nid)
    assert [(r["actor"], r["action"]) for r in rows] == [
        ("agent", "drafted"), ("agent", "requested_signoff"), ("human:Dr. Lee", "approved")]
    assert rows[-1]["detail"]["code_review"] is True  # the flags as they were when approved
    assert rows[-1]["detail"]["reason"] == "Verified against the chart."
