"""SQLite store for draft notes and the audit log.

Two tables. `notes` is written once when a draft is created; only its status and
review fields change afterwards, never the note content. `audit` is append-only:
every state change adds a row and nothing is updated or deleted.
"""
import json
import sqlite3
import uuid
from datetime import UTC, datetime

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
    reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    note_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,             -- "agent" or "human:<name>"
    action TEXT NOT NULL,            -- drafted | requested_signoff | approved | rejected
    detail TEXT NOT NULL             -- JSON: flags at the time, reason, etc.
);
-- Enforced by the database, so a bug elsewhere cannot rewrite history or edit a finished note.
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
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
            from pathlib import Path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

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
        if row is None:
            return None
        d = dict(row)
        d["note"] = json.loads(d.pop("note_json"))
        d["warnings"] = json.loads(d.pop("warnings_json"))
        d["code_review"] = bool(d["code_review"])
        d["content_review"] = bool(d["content_review"])
        return d

    def list(self, status=None):
        sql, args = "SELECT id, created_at, status, code_review, content_review FROM notes", ()
        if status:
            sql, args = sql + " WHERE status = ?", (status,)
        return [dict(r) for r in self._db.execute(sql + " ORDER BY created_at, rowid", args)]

    def set_status(self, note_id, status, actor, action, detail, reviewer=None, reason=None):
        """Change a note's status and write the audit row in one transaction."""
        reviewed = status in ("approved", "rejected")
        with self._db:
            self._db.execute(
                "UPDATE notes SET status = ?, reviewed_by = ?, review_reason = ?, reviewed_at = ? WHERE id = ?",
                (status, reviewer if reviewed else None, reason if reviewed else None,
                 _now() if reviewed else None, note_id))
            self._audit(note_id, actor, action, detail)

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
