# CliniFlow

An ambient clinical documentation assistant. It turns a doctor-patient transcript into a structured SOAP note (Subjective, Objective, Assessment, Plan) plus a suggested ICD-10 code. A validation gateway checks and repairs the model's output, and **a human has to sign off before a note counts**.

> **Built with Llama.** Fine-tuned from Llama 3.2 3B Instruct; see [NOTICE](NOTICE) for the license and data attributions.

**Scope honesty statement:** this is a portfolio project built on synthetic data, for engineering demonstration. It is **not** clinically validated, **not** a certified medical device, and **not** HIPAA-compliant infrastructure. No real patient data is used or stored anywhere in this repo. Suggested codes are never auto-submitted and require human sign-off.

The interesting part is not the model, it is the engineering around a model that is imperfect: measuring exactly where it fails, putting a checking layer in front of it, and keeping a person in charge.

## Results

Measured on 50 held-out transcripts, one T4 GPU, `temperature` 0. Full method, caveats and examples: [`evals/BENCHMARK_REPORT.md`](evals/BENCHMARK_REPORT.md) (new to the terms? read [`evals/GLOSSARY.md`](evals/GLOSSARY.md)).

| | base model | fine-tuned (adapter) | fine-tuned (merged) |
|---|---|---|---|
| Output is bare, schema-valid JSON | 0/49 (wraps it in markdown fences) | **50/50** | 49/50 |
| Exact ICD-10 code, raw | 0/49 | 0/50 | 0/50 |
| Exact ICD-10 code after the gateway's repair | n/a | **11/50** | 11/50 |
| Notes with a major unsupported claim (LLM judge) | 4/49 | 9/50 | 9/50 |
| Median request time | 14.1 s | 51.2 s | **25.2 s** |

What the numbers say:
- **Fine-tuning fixed the format** (0 to 50 of 50), which is what it was for.
- **ICD-10 code numbers stayed wrong.** The model learns diagnosis *names* much better than code numbers, so the gateway looks the code up from the model's own description (0 to 11 of 50 exact). Every repair is flagged for review, because half of them are still wrong.
- **Hallucination copies the training data.** The model invents patient details at the same rate as the reference notes. Judging all 901 training notes showed 19% contain a major claim the conversation does not support, and only 5.5% are fully supported. The cause is how the source dataset was built: notes were written first and dialogues generated *from* them, so training on dialogue to note asks the model to recover details that were never said.
- **Merging the adapter into the base weights** made serving 2.0x faster with equivalent quality (one format failure in 50, row 28, which the gateway's retry would catch). The original design kept the adapter unmerged to share one base between several adapters; with a single adapter that bought nothing.

## How it fits together

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
 vLLM on Modal (T4): the merged model ◄── built from the adapter by serving/modal_merge.py

 Offline:  MedSynth data ─► prep, noise, split ─► QLoRA training (Colab) ─► adapter registry (S3)
           evals/: Python scripts + Promptfoo, LLM judge, gateway replay
```

Each piece owns one job. The gateway never edits note text and never invents a code; it flags. The agent tools cannot approve anything. The registry keeps the small unmerged adapter as the source of truth, and the merged serving copy is derived from it.

| Pillar | What it is | Status |
|---|---|---|
| 1 Fine-tuning | QLoRA adapter (Llama 3.2 3B) on synthetic encounters, 89 minutes on a T4 | Done: [`training/`](training) |
| 2 Data and registry | SageMaker validation job, S3 adapter registry with a manifest | Done: [`registry/`](registry) |
| 3 Serving | vLLM on Modal: merged model, plus the unmerged adapter and base for comparison | Done: [`serving/`](serving) |
| 4 Evaluation | Benchmark, hallucination checks (rules and LLM judge), Promptfoo, Langfuse | Done: [`evals/`](evals) |
| 5 Gateway | FastAPI: format retry, ICD-10 repair, grounding flags. A CI workflow is written but has not run yet | Done: [`gateway/`](gateway) |
| 6 Agents and governance | MCP server, human-only approval, audit log, feedback export | Done: [`agents/`](agents) |
| 7 Speech-to-text | Local Whisper (optional) | Not started |

## Try it

You can run the whole test suite without any accounts or GPU. The model is replaced by a fake in the tests.

```bash
pip install -r agents/requirements.txt
python -m pytest gateway agents          # 127 tests
python -m ruff check gateway agents
```

Running the real models needs a Modal account and a trained adapter, neither of which is in this repo. Setup, the deployment files and the cost notes are in [`serving/README.md`](serving/README.md), the gateway in [`gateway/README.md`](gateway/README.md) and the agent tools in [`agents/README.md`](agents/README.md). The Modal apps are configured to require proxy-auth tokens (see `serving/README.md`), and no endpoint URLs are published in this repo.

## Limitations

- **Synthetic data only.** Not clinically validated, and the dataset itself is not a source of reliable medical information.
- **Small evaluation.** 50 test rows, one run per model, one LLM judge model with a human spot-check of the major flags only. Differences of a few rows are within noise.
- **Codes remain the weak spot.** 11 of 50 exact after repair.
- **The training data teaches the model to embellish.** Measured and understood (see Results); not yet fixed. A cleaning and retraining experiment is the planned next step.
- **Slow.** 25 seconds per note on a T4, and cold starts take minutes.
- **The gateway has no authentication or rate limiting,** and the approval step's terminal check guards against accidents, not against a determined user.
- **Licensing.** The MedSynth dataset card states no license, so no rights to the data are granted here; see [NOTICE](NOTICE).

## Repo layout

```
dataset/    MedSynth download, validation, noise injection, seeded train/val/test split
training/   Colab QLoRA notebook and its generator script
registry/   SageMaker validation job, S3 adapter registry, IAM policies
serving/    vLLM on Modal (merged, adapter, base), the merge job, Docker, EC2 scripts, client and load test
evals/      benchmark, hallucination checks, Promptfoo, glossary and the written report
gateway/    FastAPI gateway, ICD-10 table, tracing, tests
agents/     MCP server, sign-off rules, review CLI, tests
```

## License

Code and documentation: MIT ([LICENSE](LICENSE)). The Llama model, the MedSynth data and dependencies keep their own terms ([NOTICE](NOTICE)).
