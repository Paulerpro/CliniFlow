import io
import json
import sqlite3

import pytest

from agents import cli, signoff
from agents.store import Store

from .conftest import FLAG_WARNING, gateway_response
from .test_cli import FakeTty


def queued(store, flagged=False):
    gr = gateway_response(code_review=flagged, warnings=[FLAG_WARNING] if flagged else [])
    nid = store.create_draft("[doctor] transcript", gr)
    signoff.request_signoff(store, nid)
    return nid


# --- final code ---

def test_reviewer_can_set_the_correct_code(store, table):
    nid = queued(store)
    note = signoff.approve(store, nid, "Dr. Lee", final_code="e11.22", table=table)
    assert note["final_code"] == "E1122"
    assert store.audit(nid)[-1]["detail"]["final_code"] == "E1122"


def test_unknown_final_code_is_refused_and_nothing_changes(store, table):
    nid = queued(store)
    with pytest.raises(signoff.SignoffError, match="not an ICD-10-CM code"):
        signoff.approve(store, nid, "Dr. Lee", final_code="Q9999", table=table)
    assert store.get(nid)["status"] == "awaiting_review"


def test_final_code_needs_the_table(store):
    nid = queued(store)
    with pytest.raises(signoff.SignoffError, match="table is needed"):
        signoff.approve(store, nid, "Dr. Lee", final_code="N809")


def test_final_code_is_frozen_once_approved(store, table):
    nid = queued(store)
    signoff.approve(store, nid, "Dr. Lee", final_code="N809", table=table)
    with pytest.raises(sqlite3.IntegrityError, match="cannot change"):
        store._db.execute("UPDATE notes SET final_code = 'Z00' WHERE id = ?", (nid,))


# --- store ---

def test_verifications_are_append_only(store):
    nid = queued(store)
    store.add_verification(nid, "m", {"unsupported_claims": [], "missing_key_facts": [], "rationale": ""})
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._db.execute("UPDATE verifications SET model = 'x'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._db.execute("DELETE FROM verifications")


def test_database_from_before_final_code_is_migrated(tmp_path):
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""CREATE TABLE notes (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, status TEXT NOT NULL,
        transcript TEXT NOT NULL, note_json TEXT NOT NULL, code_review INTEGER NOT NULL,
        content_review INTEGER NOT NULL, warnings_json TEXT NOT NULL, reviewed_by TEXT,
        review_reason TEXT, reviewed_at TEXT);""")
    old.close()
    store = Store(path)
    nid = store.create_draft("t", gateway_response())
    assert store.get(nid)["final_code"] is None


# --- CLI ---

def run(argv, store, table=None, stdin=None):
    out = io.StringIO()
    cli.run(cli.build_parser().parse_args(argv), store, stdin=stdin, stdout=out, table=table)
    return out.getvalue()


def test_cli_approve_with_code(store, table):
    nid = queued(store)
    run(["approve", nid, "--reviewer", "Dr. Lee", "--code", "N809"], store, table, FakeTty(nid + "\n"))
    assert store.get(nid)["final_code"] == "N809"


def test_cli_rejects_a_bad_code_without_changing_anything(store, table):
    nid = queued(store)
    with pytest.raises(signoff.SignoffError, match="not an ICD-10-CM code"):
        run(["approve", nid, "--reviewer", "Dr. Lee", "--code", "Q9999"], store, table, FakeTty(nid + "\n"))
    assert store.get(nid)["status"] == "awaiting_review"


def test_show_includes_the_independent_check(store):
    nid = queued(store)
    store.add_verification(nid, "m", {"unsupported_claims": [{"claim": "wrong drug", "type": "fabricated_medication",
                                                              "severity": "major"}],
                                      "missing_key_facts": ["follow-up date"], "rationale": ""})
    shown = run(["show", nid], store)
    assert "[major] fabricated_medication: wrong drug" in shown and "[missing] follow-up date" in shown
    assert "verifier_major" in shown


# --- the feedback export ---

def test_export_feedback(store, table, tmp_path):
    good = queued(store)
    signoff.approve(store, good, "Dr. Lee", final_code="E1122", table=table)
    bad = queued(store, flagged=True)
    store.add_verification(bad, "m", {"unsupported_claims": [{"claim": "invented drug", "type": "fabricated_medication",
                                                              "severity": "major"}],
                                      "missing_key_facts": [], "rationale": ""})
    signoff.reject(store, bad, "Dr. Lee", "Invented a medication in the plan.")
    queued(store)  # still awaiting review: must not appear

    out = tmp_path / "feedback.jsonl"
    msg = run(["export-feedback", "--out", str(out)], store, table)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert "2 reviewed note(s)" in msg and len(records) == 2

    approved, rejected = records
    example = approved["training_example"]
    assert example["input"] == "[doctor] transcript"
    assert example["output"]["suggested_codes"] == [{
        "code": "E1122", "system": "ICD-10", "confidence": "high",
        "description": "TYPE 2 DIABETES MELLITUS WITH DIABETIC CHRONIC KIDNEY DISEASE"}]  # the reviewer's code, dataset style
    assert approved["model_output"]["suggested_codes"][0]["code"] == "N809"  # what the model had said
    assert set(example["output"]) == {"subjective", "objective", "assessment", "plan", "suggested_codes"}

    assert rejected["status"] == "rejected" and rejected["training_example"] is None
    assert rejected["reason"] == "Invented a medication in the plan."
    assert rejected["verifier_major_claims"] == ["invented drug"]
    assert rejected["flags"]["code_review"] is True


def test_export_keeps_the_models_codes_when_the_reviewer_set_none(store, table, tmp_path):
    nid = queued(store)
    signoff.approve(store, nid, "Dr. Lee")
    out = tmp_path / "f.jsonl"
    run(["export-feedback", "--out", str(out)], store, table)
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["training_example"]["output"]["suggested_codes"][0]["code"] == "N809"
    assert "original_code" not in rec["training_example"]["output"]["suggested_codes"][0]
