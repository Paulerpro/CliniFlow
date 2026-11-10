"""Human review commands. This is the only way a note becomes approved or rejected.

    python -m agents.cli list [--status awaiting_review]
    python -m agents.cli show <note_id>
    python -m agents.cli approve <note_id> --reviewer "Dr. Name" [--reason "..."] [--code N809]
    python -m agents.cli reject  <note_id> --reviewer "Dr. Name" --reason "..."
    python -m agents.cli audit [<note_id>]
    python -m agents.cli export-feedback --out feedback.jsonl

approve and reject ask for a typed confirmation and refuse to run without a terminal, so a
script or an agent piping input cannot do it silently. This is a guard against accidents,
not a security boundary: anything that can run commands as you could still run it.
"""
import argparse
import json
import sys

from . import config, signoff
from .store import Store


def _print_note(note, out):
    flags = [k for k, v in signoff.flags(note).items() if v]
    print(f"note {note['id']}  status: {note['status']}  flags: {', '.join(flags) or 'none'}", file=out)
    for w in note["warnings"]:
        print(f"  [{w['severity']}/{w['category']}] {w['type']}: {w['detail']}", file=out)
    v = note["verification"]
    if v:
        print(f"\nindependent check ({v['model']}): {len(v['unsupported_claims'])} unsupported claim(s), "
              f"{len(v['missing_key_facts'])} missing fact(s)", file=out)
        for c in v["unsupported_claims"]:
            print(f"  [{c['severity']}] {c['type']}: {c['claim']}", file=out)
        for m in v["missing_key_facts"]:
            print(f"  [missing] {m}", file=out)
    n = note["note"]
    for field in ("subjective", "objective", "assessment", "plan"):
        print(f"\n--- {field} ---\n{n[field]}", file=out)
    print("\n--- suggested codes ---", file=out)
    for c in n["suggested_codes"]:
        print(f"  {c['code']}  {c['description']}  ({c['status']}"
              + (f", was {c['original_code']}" if c.get("original_code") else "") + ")", file=out)
    if note["reviewed_by"]:
        print(f"\nreviewed by {note['reviewed_by']}: {note['review_reason'] or '(no reason given)'}", file=out)
        if note["final_code"]:
            print(f"final code set by reviewer: {note['final_code']}", file=out)


def _confirm(action, note_id, stdin=None, stdout=None):
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    if not stdin.isatty():
        raise signoff.SignoffError(f"{action} needs an interactive terminal; refusing to run from a pipe or script")
    stdout.write(f"Type the note id ({note_id}) to confirm {action}: ")
    stdout.flush()
    if stdin.readline().strip() != note_id:
        raise signoff.SignoffError(f"{action} cancelled: the id did not match")


def feedback_records(store, table=None):
    """One record per reviewed note: what the model wrote and what the human decided."""
    records = []
    for n in store.reviewed():
        v = n["verification"]
        rec = {
            "note_id": n["id"], "status": n["status"], "reviewer": n["reviewed_by"],
            "reason": n["review_reason"], "final_code": n["final_code"],
            "flags": signoff.flags(n),
            "verifier_major_claims": [c["claim"] for c in v["unsupported_claims"] if c["severity"] == "major"] if v else [],
            "input": n["transcript"], "model_output": n["note"],
            "training_example": None,
        }
        if n["status"] == "approved":
            output = {k: n["note"][k] for k in ("subjective", "objective", "assessment", "plan")}
            codes = n["note"]["suggested_codes"]
            if n["final_code"] and table is not None:
                entry = table.lookup(n["final_code"])
                codes = [{"code": entry.code, "system": "ICD-10",
                          "description": entry.description.upper(), "confidence": "high"}]
            output["suggested_codes"] = [{k: c[k] for k in ("code", "system", "description", "confidence")}
                                         for c in codes]
            rec["training_example"] = {"input": n["transcript"], "output": output}
        records.append(rec)
    return records


def run(args, store, stdin=None, stdout=None, table=None):
    out = stdout or sys.stdout
    if args.command == "list":
        for n in store.list(args.status):
            flags = ",".join(k for k in ("code_review", "content_review") if n[k]) or "-"
            print(f"{n['id']}  {n['created_at']}  {n['status']:<16} flags: {flags}", file=out)
    elif args.command == "show":
        note = store.get(args.note_id)
        if note is None:
            raise signoff.SignoffError(f"no note with id {args.note_id}")
        _print_note(note, out)
    elif args.command == "approve":
        if args.code and table is None:
            from gateway.icd10 import IcdTable
            table = IcdTable.load()
        _confirm("approve", args.note_id, stdin, stdout)
        note = signoff.approve(store, args.note_id, args.reviewer, args.reason, args.code, table)
        print(f"approved {note['id']}", file=out)
    elif args.command == "reject":
        _confirm("reject", args.note_id, stdin, stdout)
        note = signoff.reject(store, args.note_id, args.reviewer, args.reason)
        print(f"rejected {note['id']}", file=out)
    elif args.command == "audit":
        for r in store.audit(args.note_id):
            print(f"{r['seq']:>4}  {r['ts']}  {r['note_id']}  {r['actor']:<18} {r['action']:<18} "
                  f"{json.dumps(r['detail'])}", file=out)
    elif args.command == "export-feedback":
        if table is None:
            from gateway.icd10 import IcdTable
            table = IcdTable.load()
        records = feedback_records(store, table)
        with open(args.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        approved = sum(r["training_example"] is not None for r in records)
        print(f"wrote {len(records)} reviewed note(s) to {args.out} "
              f"({approved} approved, {len(records) - approved} rejected)", file=out)


def build_parser():
    p = argparse.ArgumentParser(prog="python -m agents.cli")
    sub = p.add_subparsers(dest="command", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--status", choices=["draft", "awaiting_review", "approved", "rejected"])
    sub.add_parser("show").add_argument("note_id")
    for name in ("approve", "reject"):
        s = sub.add_parser(name)
        s.add_argument("note_id")
        s.add_argument("--reviewer", required=True)
        s.add_argument("--reason", required=(name == "reject"))
        if name == "approve":
            s.add_argument("--code", help="the correct ICD-10 code, if the note's code is wrong")
    sub.add_parser("audit").add_argument("note_id", nargs="?")
    sub.add_parser("export-feedback").add_argument("--out", required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        run(args, Store(config.DB_PATH))
    except signoff.SignoffError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
