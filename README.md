# CliniFlow

An ambient clinical documentation assistant: it turns a clinical encounter transcript into a structured SOAP note (Subjective, Objective, Assessment, Plan) plus a suggested, non-authoritative ICD-10 code, using a fine-tuned LoRA adapter served with vLLM.

**Scope honesty statement:** this is a portfolio project built on synthetic data, for engineering demonstration. It is **not** clinically validated, **not** a certified medical device, and **not** HIPAA-compliant infrastructure. No real patient data is used or stored anywhere in this repo. Suggested codes are never auto-submitted and require human sign-off.

## Status

| Pillar | What it is | Status |
|---|---|---|
| 1 &mdash; Fine-tuning | QLoRA adapter (Llama 3.2 3B) on synthetic encounter data | ✅ Done &mdash; config A trained, pushed to a private Hugging Face repo |
| 2 &mdash; AWS data & registry | SageMaker data validation + S3 adapter registry | ✅ Done &mdash; validated in the cloud, adapter registered with a manifest |
| 3 &mdash; Serving (vLLM) | Base model + LoRA adapter served over an OpenAI-style API | ✅ Done &mdash; live on Modal (T4 GPU), returning schema-valid notes over the real API; AWS EC2 deployment also scripted |
| 4 &mdash; Evaluation & telemetry | Base-vs-fine-tuned benchmark, Promptfoo/Ragas, Langfuse tracing | 🟡 Benchmark + hallucination eval done ([`evals/BENCHMARK_REPORT.md`](evals/BENCHMARK_REPORT.md)); Promptfoo/Langfuse pending |
| 5 &mdash; Software engineering | FastAPI gateway, validation + retry/repair, tests, CI | ⏳ Not started |
| 6 &mdash; Agents & governance | MCP server, human-sign-off rules | ⏳ Not started |
| 7 &mdash; Speech-to-text (optional) | Local Whisper transcription | ⏳ Not started |

See each pillar's own README for the real detail: [`dataset/`](dataset), [`training/colab/`](training/colab), [`registry/`](registry), [`serving/`](serving).

## Architecture

```
[Text input] or [Audio -> Whisper STT (optional)]
        |
        v
[Transcript]
        |
        v
[Pillar 1: QLoRA-tuned LoRA adapter]  -- NOT merged into the base model
        |
        v
[Pillar 5: FastAPI gateway]  (validation + repair layer)
        |
        v
[Pillar 3: vLLM]  -- one base model loaded once, adapter attached per request
        |
        v
[Pillar 6: MCP server + governance]  (human sign-off required)
        |
        v
[Pillar 4: Evals/telemetry]

[Pillar 2: SageMaker]  -- offline side path: dataset prep + adapter registry (S3) only
```

Each pillar owns one responsibility and nothing else &mdash; SageMaker never hosts inference, vLLM never trains, and the adapter is never merged into the base model. The full reasoning behind every one of these choices lives in the project's design document (kept locally in `docs/`, not published in this repo).

## What's actually working right now

- A 1,000-row synthetic dataset (train/val/test split), schema-validated twice: once locally and once as a real SageMaker Processing job in the cloud.
- A trained LoRA adapter (rank 16) that reliably produces schema-valid SOAP JSON: **20/20** on a held-out check.
- The same adapter served live through vLLM with LoRA enabled, on a real GPU (Colab T4) &mdash; a transcript in, a structured note out, over a normal HTTP API.
- A versioned adapter registry in S3, with a manifest recording the training config and eval scores.

## Known limitation

The trained adapter matches the output **format** perfectly (20/20 valid JSON, 20/20 schema-valid) but does not yet predict the correct ICD-10 code (0/20 on a held-out check). It is reliable at structuring a note; it is not yet reliable at suggesting the right billing code. This is flagged here deliberately rather than left to be discovered &mdash; see `registry/README.md` for the full numbers.

## Repo layout

```
dataset/    raw data + validation + noise injection + train/val/test split
training/   Colab QLoRA fine-tuning notebook and its generator script
registry/   AWS: dataset validation (SageMaker), S3 adapter registry, IAM policies
serving/    vLLM serving: config, client, load test, Docker, Modal + EC2 deployment scripts
agents/     (planned) MCP server + governance
evals/      (planned) benchmark suite
gateway/    (planned) FastAPI gateway
stt/        (planned, optional) speech-to-text
```

## Running what exists today

```bash
# Validate the dataset locally
python registry/validate_dataset.py --input-dir training/colab --output-dir registry/report

# Send a transcript to a running vLLM server (local, Colab, Modal, or EC2)
export SERVER_URL=http://<server-ip>:8000
python serving/client.py --transcript-file path/to/transcript.txt

# Load-test a running server
python serving/load_test.py --transcripts-file training/colab/val.jsonl --n 20 --concurrency 5
```

Each of these has its own README with full setup steps, since getting a GPU server running is the part with real prerequisites (quota, IAM, Docker).

## Why these specific choices

- **3B model, not 8B/7B:** fits a 16GB T4 in full precision with headroom for the KV cache; an 8B model would not.
- **LoRA adapter, never merged:** vLLM loads the base model once and attaches the adapter per request, so iterating on training doesn't mean re-uploading a multi-GB model each time.
- **SageMaker for data and registry only, vLLM for serving:** one inference path, not two competing ones.
- **No quantization by default:** keeps the fine-tune's accuracy gains intact; available later as a separately-evaluated experiment.

The full decision log, with alternatives considered and why they were rejected, is kept in the project's design document.
