# Agents and governance (Pillar 6)

An MCP server that lets an AI agent draft clinical notes, plus the human sign-off rules around them. The point is one rule: **an agent can draft a note and ask for review, but only a person can approve it.**

    agent (Claude, etc.)                         human reviewer
        |  MCP tools                                  |  python -m agents.cli
        v                                             v
    draft_note ──> gateway (Pillar 5) ──> model     list / show / approve / reject / audit
    get_note
    request_signoff ─────────> awaiting_review ──────> approved | rejected
              (SQLite: drafts + append-only audit log)

## The three tools

| Tool | What it does |
|---|---|
| `draft_note(transcript)` | Sends the transcript to the gateway, stores the result as a **draft**, returns the note, its flags and an id |
| `get_note(note_id)` | Returns a note's content, flags and status (`draft`, `awaiting_review`, `approved`, `rejected`) |
| `request_signoff(note_id)` | Moves a draft into the human review queue. It does **not** approve anything |

There is deliberately no approve tool. If the agent could approve its own note, "human sign-off" would mean nothing, and a test (`test_exactly_three_tools_and_no_way_to_approve`) fails if one is ever added.

## Human review

    python -m agents.cli list --status awaiting_review
    python -m agents.cli show <note_id>
    python -m agents.cli approve <note_id> --reviewer "Dr. Name" --reason "Checked the code against the chart."
    python -m agents.cli reject  <note_id> --reviewer "Dr. Name" --reason "Invented a medication."
    python -m agents.cli audit [<note_id>]

## Rules

- A note must be sent to review (`awaiting_review`) before it can be approved. A draft cannot be approved directly.
- A note the gateway flagged (`code_review` or `content_review`) needs a written reason of at least 10 characters to approve. Rejecting always needs one.
- `approve` and `reject` ask you to type the note id back and refuse to run without an interactive terminal.
- Every step is written to an audit log with who did it (`agent` or `human:<name>`), when, and the flags at that moment.
- The database itself enforces that the audit log is append-only, that note content never changes after drafting, that approved or rejected notes cannot change at all, and that nothing is deleted.

## Setup

    pip install -r agents/requirements.txt

Settings come from environment variables or the repo's `.env`: `GATEWAY_URL` (default `http://localhost:8080`), `GATEWAY_TIMEOUT_S` (default 600), `AGENTS_DB` (default `agents/data/signoff.db`, gitignored because it holds transcripts). The gateway must be running (`python -m uvicorn gateway.app:app --port 8080`) and pointed at a model.

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
- Single user, local SQLite, no rate limit: an agent can create many drafts.
- Transcripts are stored in the database in clear text. The project uses synthetic data only; do not point this at real patient data.
- The flags only say where to look first. The gateway catches invented drugs, doses, doctors and ICD-10 problems, but the benchmark showed it misses most wrong-pairing errors (2 of 9 notes the LLM judge found major problems in), so a human still has to read every note.

## Tests

    python -m pytest agents -q

They cover the store and its database-level rules, the sign-off state machine, the tool list, the CLI guards, and the gateway client against the real gateway app (with a fake model).
