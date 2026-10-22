# Pillar 4: evals

Base model vs. fine-tuned adapter on the 50-row held-out test set
(`training/colab/test.jsonl`).

## Files

| File | Purpose |
|---|---|
| `run_eval.py` | Sends each test transcript to a served model, scores the reply, writes a results JSON |
| `rescore.py` | Offline rescoring of two results files (no server calls); writes `rescored.json` |
| `compare.py` | Side-by-side print of two raw results files |
| `results_base.json`, `results_finetuned.json` | Raw per-row outputs and scores |
| `rescored.json` | Output of `rescore.py` |

## Reproducing

Deploy both servers (on Windows, set `PYTHONUTF8=1` or `modal deploy` crashes on a checkmark character):

    python -m modal deploy serving/modal_app.py        # fine-tuned (adapter)
    python -m modal deploy serving/modal_app_base.py   # base model, no adapter

Run the eval against each. `--adapter-dir` is only used to load `system_prompt.txt`, so both models get the identical prompt:

    python evals/run_eval.py --transcripts-file training/colab/test.jsonl \
        --adapter-dir registry/adapters/config_A --model-name base-model \
        --server-url https://paulsilas938--cliniflow-vllm-base-serve-base.us-east.modal.run \
        --out evals/results_base.json

    python evals/run_eval.py --transcripts-file training/colab/test.jsonl \
        --adapter-dir registry/adapters/config_A \
        --server-url https://paulsilas938--cliniflow-vllm-serve.us-east.modal.run \
        --out evals/results_finetuned.json

Then:

    python evals/rescore.py --test training/colab/test.jsonl \
        --base evals/results_base.json --finetuned evals/results_finetuned.json \
        --out evals/rescored.json

## How scoring works

- **Strict valid**: the reply is bare JSON and passes the Pillar 2 schema check (`registry/validate_dataset.py`).
- **Lenient valid**: same check after stripping markdown fences / surrounding prose. This separates "can't produce the schema" from "wraps JSON in a code fence".
- **Code metrics** compare `suggested_codes` to the gold output, using the lenient parse, with codes normalised (case and punctuation dropped):
  - *exact hit*: any predicted code equals a gold code
  - *category hit*: any predicted code shares the first 3 characters with a gold code
  - *description hit*: any predicted description equals a gold description
- **Latency** is wall-clock per request. Because the two models write different amounts, `rescore.py` also reports seconds per 1,000 output characters. Token counts were not recorded in the runs, so characters stand in for tokens.

## Conditions

Both runs: vLLM 0.32.0, one T4, `--dtype half`, `--max-model-len 4096`, `temperature` 0, `max_tokens` 1500, one request at a time, same system prompt.
The only server difference is the intended one: the adapter app adds `--enable-lora --max-lora-rank 16 --lora-modules ...`.
The adapter is intentionally served unmerged (see `serving/README.md`), so the fine-tuned latency below includes LoRA overhead.

## Results (50 rows)

| metric | base | fine-tuned |
|---|---|---|
| rows scored | 49 (1 network failure) | 50 |
| strict valid | 0 | 50 |
| lenient valid | 48 | 50 |
| code exact hit | 0.000 | 0.000 |
| code category hit | 0.061 | 0.240 |
| code description hit | 0.061 | 0.120 |
| avg codes predicted per row | 4.1 | 1.0 |
| latency median / mean (s) | 14.1 / 18.0 | 51.2 / 57.0 |
| latency p90 / max (s) | 17.6 / 199.3 | 64.9 / 316.9 |
| median output chars | 1,552 | 3,061 |
| median s per 1k chars | 8.9 | 16.7 |

### Reading the results

- **Format**: the fine-tuned model returned valid bare JSON on 50/50 rows. The base model wrapped every reply in markdown fences (0/50 strict) and was schema-valid on 48/49 once fences were stripped.
- **Codes**: neither model got any exact code. The fine-tuned model often produces the right *description* with the wrong *code number* (6 of 50 rows, e.g. "ENDOMETRIOSIS, UNSPECIFIED" with N83 where the gold is N809), so it has learned the diagnosis names better than the code numbers. The base model emits about 4 codes per row, scattershot.
- **Latency**: the fine-tuned model is about 3.6x slower at the median. About 2x of that is longer output; the rest (about 1.9x per 1k characters) is consistent with unmerged-LoRA overhead on a T4. The 199s and 316s maxima are probably cold-start or container events, so prefer the medians.

## Caveats

- One base row failed with a DNS error (`getaddrinfo failed`); it is counted as invalid in the raw `results_base.json` summary and excluded from `rescored.json`.
- The Dockerfile was `FROM vllm/vllm-openai:latest` when these runs were made. Both servers were on 0.32.0 (checked via `/version`); the Dockerfile is now pinned to `v0.32.0`.
- n = 50, so differences of a few rows are not meaningful.
- Gold codes are synthetic-dataset labels, not clinical ground truth. They are all ICD-10-CM shaped (dot-less, e.g. `Z20822` is Z20.822).

## Findings for later pillars

**ICD-10 code numbers are the weak spot.** The fine-tuned model reliably produces valid JSON and often the right diagnosis *name*, but rarely the right code *number* (0/50 exact, 12% description hit, 24% category hit). Suggested handling in the gateway pillar (retry/repair), in priority order. None of this has been built or tested:

1. **Validate against an ICD-10-CM table.** Check that each returned code exists and that the table's description for it matches the one the model wrote.
2. **Repair by description lookup.** On a mismatch, take the code from the table using the model's description (fuzzy matching, since exact description matches were only 6/50), or retry the request. Wrong descriptions will still map to wrong codes, so measure this against the gold set before relying on it.
3. **Constrained decoding (optional).** vLLM can force output into a JSON schema or regex. A shape regex only guarantees well-formed codes, which the model already produces, so it would not fix wrong numbers. An allow-list of real codes (about 70k) is heavy and still lets the model pick the wrong real code.
4. **Retrain with more code-focused data (Pillar 1)**, only if the gateway fixes prove insufficient.

Re-run `rescore.py` on the repaired outputs to measure the gain with the same metrics.
