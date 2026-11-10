"""SQLite store for draft notes, independent checks and the audit log.

`notes` is written once when a draft is created; only its status and review fields change
afterwards, never the note content. `verifications` and `audit` are append-only: every
check or state change adds a row and nothing is updated or deleted.
"""
import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,            -- draft | awaiting_review | approved | rejected
    transcript TEXT NOT NULL,
    note_json TEXT NOT NULL,         -- SOAP fields + suggested_codes, exactly as the gateway returned them
    code_review INTEGER NOT NULL,
    content_review INTEGER NOT NULL,
    warnings_json TEXT NOT NULL,
    reviewed_by TEXT,
    review_reason TEXT,
    reviewed_at TEXT,
    final_code TEXT                  -- the ICD-10 code the reviewer says is right, if they set one
);
CREATE TABLE IF NOT EXISTS verifications (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    model TEXT NOT NULL,
    result_json TEXT NOT NULL        -- unsupported_claims, missing_key_facts, rationale
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,             -- "agent" or "human:<name>"
    action TEXT NOT NULL,            -- drafted | verified | requested_signoff | approved | rejected
    detail TEXT NOT NULL             -- JSON: flags at the time, reason, etc.
);
-- Enforced by the database, so a bug elsewhere cannot rewrite history or edit a finished note.
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS verifications_no_update BEFORE UPDATE ON verifications
BEGIN SELECT RAISE(ABORT, 'verifications are append-only'); END;
CREATE TRIGGER IF NOT EXISTS verifications_no_delete BEFORE DELETE ON verifications
BEGIN SELECT RAISE(ABORT, 'verifications are append-only'); END;
CREATE TRIGGER IF NOT EXISTS notes_final_is_final BEFORE UPDATE ON notes
WHEN OLD.status IN ('approved', 'rejected')
BEGIN SELECT RAISE(ABORT, 'an approved or rejected note cannot change'); END;
CREATE TRIGGER IF NOT EXISTS notes_no_delete BEFORE DELETE ON notes
BEGIN SELECT RAISE(ABORT, 'notes cannot be deleted'); END;
CREATE TRIGGER IF NOT EXISTS notes_content_fixed BEFORE UPDATE OF id, created_at, transcript, note_json,
    code_review, content_review, warnings_json ON notes
BEGIN SELECT RAISE(ABORT, 'note content cannot be edited'); END;
"""


def _now():
    return datetime.now(UTC).isoformat(timespec="seconds")


class Store:
    def __init__(self, path):
        path = str(path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self):
        """Databases made before final_code existed get the column added."""
        cols = {r["name"] for r in self._db.execute("PRAGMA table_info(notes)")}
        if "final_code" not in cols:
            with self._db:
                self._db.execute("ALTER TABLE notes ADD COLUMN final_code TEXT")

    def create_draft(self, transcript, gateway_response):
        """Store a gateway response as a new draft and return its id."""
        note_id = uuid.uuid4().hex[:12]
        gr = gateway_response
        note = {k: gr[k] for k in ("subjective", "objective", "assessment", "plan", "suggested_codes")}
        with self._db:
            self._db.execute(
                "INSERT INTO notes (id, created_at, status, transcript, note_json, code_review,"
                " content_review, warnings_json) VALUES (?, ?, 'draft', ?, ?, ?, ?, ?)",
                (note_id, _now(), transcript, json.dumps(note), int(gr["code_review"]),
                 int(gr["content_review"]), json.dumps(gr["warnings"])))
            self._audit(note_id, "agent", "drafted", {
                "code_review": gr["code_review"], "content_review": gr["content_review"],
                "warning_types": [w["type"] for w in gr["warnings"]]})
        return note_id

    def get(self, note_id):
        row = self._db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return self._note(row) if row else None

    def _note(self, row):
        d = dict(row)
        d["note"] = json.loads(d.pop("note_json"))
        d["warnings"] = json.loads(d.pop("warnings_json"))
        d["code_review"] = bool(d["code_review"])
        d["content_review"] = bool(d["content_review"])
        d["verification"] = self.latest_verification(d["id"])
        return d

    def list(self, status=None):
        sql, args = "SELECT id, created_at, status, code_review, content_review FROM notes", ()
        if status:
            sql, args = sql + " WHERE status = ?", (status,)
        return [dict(r) for r in self._db.execute(sql + " ORDER BY created_at, rowid", args)]

    def reviewed(self):
        """Every approved or rejected note, oldest first (for the feedback export)."""
        rows = self._db.execute(
            "SELECT * FROM notes WHERE status IN ('approved', 'rejected') ORDER BY reviewed_at, rowid")
        return [self._note(r) for r in rows]

    def set_status(self, note_id, status, actor, action, detail, reviewer=None, reason=None, final_code=None):
        """Change a note's status and write the audit row in one transaction."""
        reviewed = status in ("approved", "rejected")
        with self._db:
            self._db.execute(
                "UPDATE notes SET status = ?, reviewed_by = ?, review_reason = ?, reviewed_at = ?,"
                " final_code = ? WHERE id = ?",
                (status, reviewer if reviewed else None, reason if reviewed else None,
                 _now() if reviewed else None, final_code if reviewed else None, note_id))
            self._audit(note_id, actor, action, detail)

    def add_verification(self, note_id, model, result):
        with self._db:
            self._db.execute("INSERT INTO verifications (note_id, ts, model, result_json) VALUES (?, ?, ?, ?)",
                             (note_id, _now(), model, json.dumps(result)))
            claims = result["unsupported_claims"]
            self._audit(note_id, "agent", "verified", {
                "model": model, "major_claims": sum(c["severity"] == "major" for c in claims),
                "minor_claims": sum(c["severity"] == "minor" for c in claims),
                "missing_key_facts": len(result["missing_key_facts"])})

    def latest_verification(self, note_id):
        row = self._db.execute(
            "SELECT ts, model, result_json FROM verifications WHERE note_id = ? ORDER BY seq DESC LIMIT 1",
            (note_id,)).fetchone()
        if row is None:
            return None
        return {"ts": row["ts"], "model": row["model"], **json.loads(row["result_json"])}

    def audit(self, note_id=None):
        sql, args = "SELECT seq, note_id, ts, actor, action, detail FROM audit", ()
        if note_id:
            sql, args = sql + " WHERE note_id = ?", (note_id,)
        rows = [dict(r) for r in self._db.execute(sql + " ORDER BY seq", args)]
        for r in rows:
            r["detail"] = json.loads(r["detail"])
        return rows

    def _audit(self, note_id, actor, action, detail):
        self._db.execute("INSERT INTO audit (note_id, ts, actor, action, detail) VALUES (?, ?, ?, ?, ?)",
                         (note_id, _now(), actor, action, json.dumps(detail)))
