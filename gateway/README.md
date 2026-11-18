# Gateway (Pillar 5)

A small FastAPI service in front of the vLLM server. It takes a transcript and returns a SOAP note that has been **format-checked, retried if unusable, code-repaired and flagged**. It exists because the benchmark ([`evals/BENCHMARK_REPORT.md`](../evals/BENCHMARK_REPORT.md)) showed what the model is good and bad at:

| Model weakness (measured) | What the gateway does |
|---|---|
| Output format is reliable (50/50) but not guaranteed | Strict JSON + schema check; re-asks up to `MAX_RETRIES` times, showing the model its bad reply |
| Right diagnosis name, wrong ICD-10 number (0/50 exact) | Looks the code up in the ICD-10-CM table; if wrong, replaces it using the model's own description |
| Invents drugs, doses, vitals, referral names | Checks them against the transcript and flags the note |
| Invents patient age, ethnicity, name (about 49/50 notes) | Recorded as an `info` warning only; it is too common to flag for review |

## Run

    pip install -r gateway/requirements.txt
    python -m uvicorn gateway.app:app --port 8080

Point it at a model server with `VLLM_URL`, either as an environment variable or as a line in the repo's `.env` file (only gateway settings are read from `.env`; real environment variables win). Open http://localhost:8080/health to confirm which server it is using, and http://localhost:8080/docs to try `POST /v1/notes` in the browser. A cold Modal container can take several minutes to answer the first request.

    curl -X POST localhost:8080/v1/notes -H "Content-Type: application/json" -d '{"transcript": "[doctor] ... [patient] ..."}'

Tests (offline, fake model): `python -m pytest gateway` · Lint: `python -m ruff check gateway`

## Response

`POST /v1/notes` returns the four SOAP fields, `suggested_codes`, and:

- `code_review`: a code was repaired, did not match its description, or does not exist
- `content_review`: the note states a drug, dose, measured value or doctor the transcript does not
- `needs_review`: either of the above
- `warnings`: each has a `type`, a `severity` (`review` sets the flags above, `info` is recorded only), a `category` (`code` or `content`) and a `detail`
- `attempts`, `model`, `latency_ms`

Each code carries a `status`: `verified`, `verified_non_billable` (a real parent code such as N508, accepted with an info warning), `repaired` (the original is kept in `original_code`), `unresolved_mismatch` or `unresolved_not_found` (the model's code is kept and flagged; the gateway never silently drops or invents a code).

Errors: `502` if the model never returned a usable reply (with the attempt count and problems), `503` if the model server is unreachable or times out, `422` for an empty or oversized transcript.

## Tracing (Langfuse, optional)

Tracing is off until you set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` (and `LANGFUSE_HOST` for a non-default region, e.g. `https://us.cloud.langfuse.com`), either as environment variables or in `.env`. `/health` shows `"tracing": true` when it is on. Each request becomes one trace: a root span `generate_note`, one `llm_call` generation per model attempt (retries show up separately, with the format problem), then `repair_codes` and `grounding`. Scores on the trace: `needs_review`, `code_review`, `content_review`, `attempts`, `codes_repaired`.

By default **no note text is sent**: traces carry sizes, flags, statuses and scores only. Set `TRACE_CONTENT=true` to include transcripts and notes (synthetic data only). Tracing can never break a request: errors from the Langfuse client are logged and ignored, and the test suite clears the Langfuse keys so a test run cannot send traces to your real project.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_URL` | `http://localhost:8000` | OpenAI-compatible model server |
| `MODEL_NAME` | `clinical-json-extractor-merged` | `model` field sent to the server (`clinical-json-extractor` for the unmerged adapter app) |
| `SYSTEM_PROMPT_PATH` | `registry/adapters/config_A/system_prompt.txt` | must match the prompt the adapter was trained with |
| `MAX_RETRIES` | `2` | re-asks after an unusable reply |
| `ALLOW_LENIENT_PARSE` | `false` | accept replies wrapped in a markdown fence (the benchmark found the fine-tuned model never needs this) |
| `REPAIR_THRESHOLD` | `90` | minimum description similarity (0 to 100) to replace a code |
| `REQUEST_TIMEOUT_S` | `600` | a cold Modal container can take minutes to answer |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | unset | enable tracing (see above) |
| `TRACE_CONTENT` | `false` | include note text in traces |

## Measured effect

Replaying the 50 saved fine-tuned outputs through the real pipeline (`python evals/gateway_replay.py ...`, no model or API calls; the threshold was fixed at 90 before running):

| | before gateway | after gateway |
|---|---|---|
| Exact ICD-10 match | 0/50 | **11/50** |
| Same 3-character category | 12/50 | **26/50** |
| Diagnosis description match | 6/50 | 11/50 |

Review flags on the same 50 notes: `code_review` 47, `content_review` 2. Both `content_review` notes are ones the LLM judge also marked as containing a major unsupported claim, but the content check catches only 2 of the judge's 9 such notes. It finds invented numbers, drugs, doses and doctor names; it cannot see a wrong drug-condition pairing such as "hypertension managed with Metformin".

## Limits

- `code_review` is true for almost every note (47/50): the model's code and description almost never agree with the table, and a repaired code is still only the table entry for the model's own description. Repairs are right 11 times out of 20 at the default threshold, so they are flagged, never trusted silently.
- The content check is precise but low-recall (2 of 9). Human sign-off (Pillar 6) is still required; the flags only say where to look first.
- The gateway checks every code in `suggested_codes` independently; it does not choose among them or limit how many are returned.
- The ICD-10-CM table is the CDC/NCHS FY2026 order file (public domain; 98,186 entries, 74,719 billable) in `data/icd10cm_fy2026.tsv.gz`. FY2027 was not yet published at the time of writing; update the file each October.
- The gateway re-implements the schema and grounding checks from `registry/` and `evals/` on purpose, so it can ship on its own. If you change one, change the other.
