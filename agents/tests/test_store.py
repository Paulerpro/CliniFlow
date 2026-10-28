import sqlite3

import pytest

from .conftest import FLAG_WARNING, gateway_response


def draft(store, **kw):
    return store.create_draft("transcript text", gateway_response(**kw))


def test_draft_round_trip(store):
    nid = draft(store, code_review=True, warnings=[FLAG_WARNING])
    note = store.get(nid)
    assert note["status"] == "draft" and note["code_review"] is True and note["content_review"] is False
    assert note["note"]["suggested_codes"][0]["code"] == "N809"
    assert note["warnings"][0]["type"] == "code_repaired"
    assert note["transcript"] == "transcript text"


def test_unknown_id_is_none(store):
    assert store.get("nope") is None


def test_drafting_is_audited(store):
    nid = draft(store, code_review=True, warnings=[FLAG_WARNING])
    (row,) = store.audit(nid)
    assert row["actor"] == "agent" and row["action"] == "drafted"
    assert row["detail"]["warning_types"] == ["code_repaired"]


def test_list_filters_by_status(store):
    a, b = draft(store), draft(store)
    store.set_status(b, "awaiting_review", "agent", "requested_signoff", {})
    assert [n["id"] for n in store.list("awaiting_review")] == [b]
    assert {n["id"] for n in store.list()} == {a, b}


def test_audit_log_is_append_only(store):
    nid = draft(store)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._db.execute("UPDATE audit SET actor = 'human:me' WHERE note_id = ?", (nid,))
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._db.execute("DELETE FROM audit")


def test_note_content_cannot_be_edited(store):
    nid = draft(store)
    with pytest.raises(sqlite3.IntegrityError, match="content cannot be edited"):
        store._db.execute("UPDATE notes SET note_json = '{}' WHERE id = ?", (nid,))
    with pytest.raises(sqlite3.IntegrityError, match="content cannot be edited"):
        store._db.execute("UPDATE notes SET code_review = 0 WHERE id = ?", (nid,))


def test_notes_cannot_be_deleted(store):
    draft(store)
    with pytest.raises(sqlite3.IntegrityError, match="cannot be deleted"):
        store._db.execute("DELETE FROM notes")


def test_finished_note_cannot_change_at_all(store):
    nid = draft(store)
    store.set_status(nid, "approved", "human:dr", "approved", {}, reviewer="dr", reason="ok")
    with pytest.raises(sqlite3.IntegrityError, match="cannot change"):
        store.set_status(nid, "draft", "agent", "x", {})
