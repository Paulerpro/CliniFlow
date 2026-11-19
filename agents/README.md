# Agents and governance (Pillar 6)

An MCP server that lets an AI agent draft clinical notes, plus the human sign-off rules around them. The point is one rule: **an agent can draft a note and ask for review, but only a person can approve it.**

    agent (Claude, etc.)                         human reviewer
        |  MCP tools                                  |  python -m agents.cli
        v                                             v
    draft_note ──> gateway (Pillar 5) ──> model     list / show / approve / reject / audit
    get_note
    request_signoff ─────────> awaiting_review ──────> approved | rejected
              (SQLite: drafts + append-only audit log)

## The five tools

| Tool | What it does |
|---|---|
| `draft_note(transcript)` | Sends the transcript to the gateway, stores the result as a **draft**, returns the note, its flags and an id |
| `get_note(note_id)` | Returns a note's content, flags, independent-check results and status (`draft`, `awaiting_review`, `approved`, `rejected`) |
| `verify_note(note_id)` | Runs an independent second check by a different model (OpenAI, `gpt-4.1-mini` by default): lists claims the transcript does not support (major or minor) and key facts the note omits. A **major** finding means a reviewer must write a reason to approve. Needs `OPENAI_API_KEY`; sends the transcript and note to OpenAI |
| `search_icd10(query, limit)` | Searches the official ICD-10-CM list by diagnosis wording. Read-only; use it to suggest candidate codes to the reviewer |
| `request_signoff(note_id)` | Moves a draft into the human review queue. It does **not** approve anything |

There is deliberately no approve tool. If the agent could approve its own note, "human sign-off" would mean nothing, and a test (`test_exactly_five_tools_and_no_way_to_approve`) fails if one is ever added. The agent also cannot edit a note or change its codes: notes are immutable, and corrections are made by the human at approval time.

`verify_note` is a second opinion, not proof. It is one model and one pass. On the benchmark's judge runs it flagged the notes with wrong drug-condition pairings that the gateway's own checks cannot see, but it can be wrong in both directions, so the reviewer still reads the note.

## Human review

    python -m agents.cli list --status awaiting_review
    python -m agents.cli show <note_id>
    python -m agents.cli approve <note_id> --reviewer "Dr. Name" --reason "Checked the code against the chart." [--code N809]
    python -m agents.cli reject  <note_id> --reviewer "Dr. Name" --reason "Invented a medication."
    python -m agents.cli audit [<note_id>]
    python -m agents.cli export-feedback --out feedback.jsonl

## Rules

- A note must be sent to review (`awaiting_review`) before it can be approved. A draft cannot be approved directly.
- A note needs a written reason of at least 10 characters to approve if the gateway flagged it (`code_review` or `content_review`) or the independent check found a major claim (`verifier_major`). Rejecting always needs one.
- `approve --code N809` records the code the reviewer says is right (it must exist in the ICD-10-CM list). The note itself is never edited.
- `approve` and `reject` ask you to type the note id back and refuse to run without an interactive terminal.
- Every step is written to an audit log with who did it (`agent` or `human:<name>`), when, and the flags at that moment.
- The database itself enforces that the audit log is append-only, that note content never changes after drafting, that approved or rejected notes cannot change at all, and that nothing is deleted.

## The feedback loop

`export-feedback` writes one JSON line per approved or rejected note: the transcript, what the model wrote, the reviewer's decision and reason, the flags, the verifier's major claims, and for approved notes a `training_example` in the dataset's format (`{input, output}`) with the reviewer's code swapped in. Rejected notes (with reasons) are ready-made regression cases for the evals. This is how reviewer corrections flow back into the next fine-tune (Pillar 1) and into the eval suite.

Be aware of what it does **not** fix: reviewers correct codes and accept or reject, but they cannot edit text. Invented demographics ("a 45-year-old African American male") sit in most notes as an `info` warning and would survive into an approved `training_example`, so a cleaning step is needed before using the export for training. Do not train on the export without reading a sample of it.

## Setup

    pip install -r agents/requirements.txt

Settings come from environment variables or the repo's `.env`: `GATEWAY_URL` (default `http://localhost:8080`), `GATEWAY_TIMEOUT_S` (default 600), `AGENTS_DB` (default `agents/data/signoff.db`, gitignored because it holds transcripts), `OPENAI_API_KEY` and `VERIFIER_MODEL` (default `gpt-4.1-mini`) for `verify_note`. Langfuse tracing uses the gateway's settings (see `gateway/README.md`); `verify_note` is traced too. The gateway must be running (`python -m uvicorn gateway.app:app --port 8080`) and pointed at a model.

### Try it without a GPU

The agent layer only needs a gateway to talk to, and the gateway has a demo mode. In one terminal run `python -m gateway.demo`; in another, use the tools from any MCP client, or run the review side yourself with the CLI below. `verify_note` is the one tool that needs an OpenAI key (it calls a second model); the others work as they are.

### Connect an MCP client

Claude Code, from the repo root:

    claude mcp add cliniflow -- python -m agents.mcp_server

Claude Desktop (`claude_desktop_config.json`), with `PYTHONPATH` so it works from any folder:

    {
      "mcpServers": {
        "cliniflow": {
          "command": "python",
          "args": ["-m", "agents.mcp_server"],
          "env": { "PYTHONPATH": "C:\\Users\\HP\\projects\\AI\\CliniFlow" }
        }
      }
    }

A cold model container can take several minutes to answer the first `draft_note`, which may exceed the client's own tool timeout. Warm it up first (open the gateway's `/docs` and send one request).

## Limits (read these)

- **The approval guard is not a security boundary.** The terminal check and typed confirmation prevent accidents and scripted approvals. Anything that can run commands as you, including an agent with shell access, could still run the CLI. If the agent has a shell, keep it away from this repo's database or run review on a separate machine.
- **Reviewer names are self-declared.** There is no login. The audit log records the name that was typed.
- Single user, local SQLite, no rate limit: an agent can create many drafts, and each `verify_note` call costs a fraction of a cent.
- `search_icd10` works best with clinical diagnosis phrases ("type 2 diabetes with diabetic chronic kidney disease"). Plain symptom words ("sore throat") rank poorly.
- Transcripts are stored in the database in clear text. The project uses synthetic data only; do not point this at real patient data.
- The flags only say where to look first. The gateway catches invented drugs, doses, doctors and ICD-10 problems, but the benchmark showed it misses most wrong-pairing errors (2 of 9 notes the LLM judge found major problems in), so a human still has to read every note.

## Tests

    python -m pytest agents -q

They cover the store and its database-level rules, the sign-off state machine, the tool list, the CLI guards, the verifier (against a mock of the OpenAI API, never the real one), the feedback export, and the gateway client against the real gateway app (with a fake model).
