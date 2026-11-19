# Architecture

CliniFlow turns a doctor-patient conversation into a SOAP note with a suggested ICD-10 code, using a small fine-tuned model, and treats that model as an imperfect component: every layer around it exists because something specific was measured to go wrong. This document explains each layer, why it is there, and the evidence behind the design. The README has the headline results; per-folder READMEs have the commands.

Suggested reading order: [README](README.md), this file, [`evals/BENCHMARK_REPORT.md`](evals/BENCHMARK_REPORT.md), then [`gateway/`](gateway/README.md) and [`agents/`](agents/README.md).

## One note, end to end

1. An AI agent (any MCP client) calls `draft_note` with a transcript. The tool sends it to the **gateway**.
2. The gateway asks the **model** (vLLM on Modal) for a note and checks the reply: it must be bare, schema-valid JSON, or the model is shown its mistake and asked again (up to two retries).
3. Each suggested code is looked up in the official ICD-10-CM table. A code that doesn't match its own description is replaced using the description, and flagged. Drugs, doses, measured values and doctor names in the note are checked against the transcript, and flagged if the conversation never mentions them.
4. The result is stored as a **draft** with its flags. The agent can ask for an independent second check (`verify_note`), look up candidate codes (`search_icd10`), and hand the draft to review (`request_signoff`). It cannot approve anything.
5. A person runs the review CLI, reads the note and its flags, and approves or rejects it, giving a written reason whenever anything was flagged. Every step is written to an append-only audit log, and approved or rejected notes are frozen by the database itself.
6. Reviewers' corrections can be exported as training and regression data for the next round.

A real example from a held-out conversation: the model returned `N83` with the description "ENDOMETRIOSIS, UNSPECIFIED". The gateway saw that the code and description disagree, looked the description up, replaced the code with `N809` (the correct one) and set `code_review` so a human checks it.

```
 AI agent (MCP client)            human reviewer
   draft_note · get_note            python -m agents.cli
   verify_note · search_icd10         list · show · approve · reject
   request_signoff                    audit · export-feedback
        │                                   │
        ▼                                   ▼
 ┌─────────────────────────┐     SQLite: drafts, checks, append-only audit log
 │ Gateway (FastAPI)       │
 │  strict JSON + retry    │──► Langfuse traces (optional)
 │  ICD-10 table repair    │
 │  grounding flags        │
 └───────────┬─────────────┘
             ▼
 vLLM on Modal (T4): the CliniFlow model, served as cliniflow-soap
```

## The layers, and why each exists

**Data** ([`dataset/`](dataset/README.md)). MedSynth, a public set of synthetic dialogue and SOAP-note pairs, sampled to 1,000 rows, parsed, given seeded noise on the dialogue side and split 901 / 49 / 50 (the 50 are the held-out test set behind every result). The data is not in the repo because its license is unstated; a manifest of hashes verifies a regeneration is exact. *Finding that shaped everything after:* MedSynth wrote each note first and generated the conversation from it, so training conversation-to-note asks the model to recover details the conversation never states. Judging all 901 training notes found 19% contain a major unsupported claim and only 5.5% are fully supported.

**Training** ([`training/`](training)). QLoRA on Llama 3.2 3B Instruct (rank 16, alpha 32, 2 epochs, loss on the note only), 89 minutes on a T4. Training produces a small adapter, not a new model. It learned the format perfectly and the ICD-10 code numbers poorly (0 of 50 exact), which is why the gateway exists.

**Registry** ([`registry/`](registry/README.md)). A SageMaker Processing job validates the data, and the adapter is stored in S3 with a manifest of its training settings. SageMaker never hosts the model: there is one inference path, not two.

**Serving** ([`serving/`](serving/README.md)). The adapter is merged into the fp16 base weights by a Modal job that checks its own result (three weights recomputed by hand, logits compared) before saving, then served as plain vLLM. Merging was a reversal: the original design served the adapter unmerged so several adapters could share one base model. With a single adapter that bought nothing and cost 2x in speed (median 51 s to 25 s per note, equivalent quality). The unmerged adapter and the plain base model remain in `serving/experiments/` as reference points for the benchmark.

**Gateway** ([`gateway/`](gateway/README.md)). A thin FastAPI service. It checks, repairs where it can be sure, and flags everything else; it **never edits note text and never invents a code**. Its ICD-10 table is the CDC FY2026 list (98,186 entries, 74,719 billable). On the saved outputs, repair lifted exact code matches from 0 to 11 of 50. A repair is refused when the two descriptions disagree on a word that flips a diagnosis (male/female, left/right, with/without, acute/chronic): plain string similarity treats "FEMALE" and "MALE" as nearly identical, and building the demo caught the gateway turning a woman's diagnosis into a man's. Flags come in two kinds: `code_review` (a code was repaired or doesn't exist) and `content_review` (the note states a drug, dose, value or doctor the conversation doesn't). Invented patient demographics are recorded as information only, because they appear in nearly every note and flagging them would make the flag meaningless. The gateway adds about a tenth of a second to a request that takes 25 seconds in the model.

**Agents and governance** ([`agents/`](agents/README.md)). An MCP server with five tools and, deliberately, no approve tool. If an agent could approve its own output, "human sign-off" would mean nothing, so approval is a separate command only a person runs, with a typed confirmation and a written reason of at least ten characters whenever the gateway or the independent check flagged the note. The database enforces what the code promises: the audit log can't be rewritten, note content can't change after drafting, finished notes are final, nothing is deleted. `verify_note` is a second model's claim-by-claim check against the transcript; it catches things the gateway's rules cannot, such as a diabetes drug listed as the treatment for hypertension.

**Evaluation** ([`evals/`](evals/BENCHMARK_REPORT.md)). Format and code checks are deterministic. Hallucination is measured two ways: rule checks (numbers, drug and dose pairs, doctor names must appear in the transcript) and an LLM judge that lists unsupported claims by severity. The reference notes are judged too, as a control, because they embellish as well. Promptfoo reproduces the deterministic checks as a grid and agreed with the Python scripts exactly, which is how both were validated. Langfuse traces each gateway request (model call, retries, repairs, flags) with note text hidden by default.

## Decisions worth defending

| Decision | Alternative | Why this one |
|---|---|---|
| Gateway flags, never edits | Auto-correct the note | In a clinical setting a silent fix is worse than a visible flag; repaired codes are right about half the time |
| Repair codes by description lookup | Constrain the model's decoding | The model knows diagnosis names better than numbers; a shape constraint wouldn't fix a wrong number |
| Strict JSON by default | Strip markdown fences | The fine-tuned model never needs it; leniency is an opt-in setting, not a default |
| Human-only approval, outside MCP | An approve tool | An agent could approve its own work |
| Merge the adapter for serving | Serve it attached | One adapter; measured 2.0x faster, equivalent quality |
| Don't redistribute the data | Commit it | MedSynth states no license; scripts and a hash manifest reproduce it exactly |
| Judge the reference notes too | Compare models to zero | The reference data embellishes, so zero is the wrong baseline |

## What goes wrong, and what catches it

| Failure | Caught by | Outcome |
|---|---|---|
| Reply isn't bare, schema-valid JSON | Gateway strict check | Retry with the problem shown; HTTP 502 after the third failure |
| Wrong ICD-10 number, right name | Gateway code check | Replaced from the description; `code_review` |
| Code can't be repaired, or the only match would change the meaning | Gateway | Kept as returned and flagged; never dropped or invented |
| Repair lands on a category, not a billable code | Gateway | Repaired and flagged, with a note that a more specific code is needed |
| Invented drug, dose, vital or doctor | Gateway grounding check | `content_review` |
| Wrong drug-condition pairing, contradiction | `verify_note` | A major finding forces a written reason to approve |
| Invented age, ethnicity, name | Gateway | Recorded as information; the reviewer sees it |
| Model server down or timing out | Gateway | HTTP 503 with the host that failed |
| An agent tries to approve its own note | Tool list and database rules | Not possible: no such tool, and the status rules live in the database |

## What the evaluation implies

- Fine-tuning solved format, not truth. Most of the system is about the gap between a note that looks right and one that is supported by the conversation.
- The hallucination finding traces to the training data's construction, not to model size. The cleaning-and-retraining experiment (scrub invented demographics, drop notes with major claims, retrain, re-measure) is the planned fix.
- The checks catch some of what matters, not all: the gateway's content check found 2 of the 9 notes the judge marked as containing a major claim. A human still reads every note; the flags only say where to look first.

## Limits

Synthetic data only; 50 test rows, one run per model, one judge model; 25-second latency on a T4; no authentication or rate limiting on the gateway; the review CLI's terminal check stops accidents, not a determined user. See the README for the full list.
