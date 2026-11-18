# CliniFlow benchmark: base model vs. fine-tuned adapter

**Question.** Does the QLoRA adapter (config A) make Llama-3.2-3B-Instruct better at turning a doctor-patient transcript into a SOAP JSON note with a suggested ICD-10 code?

**Short answer.** It fixes output *format* completely, it does not fix ICD-10 *code accuracy*, and it makes the model reproduce the training data's habit of *inventing patient details*. The adapter served unmerged is also about 3.6x slower per request in this setup; merging it into the base weights roughly halves that (2.0x faster, with equivalent quality apart from one format failure in 50). Format reliability is the clear win; the rest is a to-do list for the gateway (Pillar 5) and the next training round.

## Setup

| | |
|---|---|
| Test set | 50 held-out transcripts (`training/colab/test.jsonl`), never seen in training |
| Models | `unsloth/Llama-3.2-3B-Instruct` (base) vs. the same model with the config A LoRA adapter (r=16, 2 epochs) |
| Serving | vLLM 0.32.0 on one T4 (Modal), `--dtype half`, `--max-model-len 4096`; the adapter was measured both served unmerged and merged into the base weights (see section 5) |
| Requests | `temperature` 0, `max_tokens` 1500, one at a time, identical system prompt for both models |
| Only server difference | the adapter app adds `--enable-lora --max-lora-rank 16 --lora-modules ...` |
| Reference notes | the dataset's own reference notes ("gold"), scored as a control wherever a check compares a note to its transcript |

One base-model request failed with a network error (DNS), so the base model is scored on 49 rows where noted. With 50 rows, differences of a few rows are not meaningful.

## Results

### 1. Output format: a clear win for the adapter

| | base | fine-tuned |
|---|---|---|
| Strict valid (bare JSON that passes the schema) | 0/50 | **50/50** |
| Lenient valid (markdown fences stripped first) | 48/49 | **50/50** |

The base model understands the schema but wraps every reply in a markdown code fence, so a strict parser rejects all of it. The adapter returns bare, schema-valid JSON every time.

### 2. ICD-10 codes: the adapter's main weakness

| | base | fine-tuned |
|---|---|---|
| Exact code match | 0.00 | 0.00 |
| Same 3-character category | 0.06 | 0.24 |
| Same diagnosis description | 0.06 | 0.12 |
| Codes predicted per note (gold has 1) | 4.1 | 1.0 |

The adapter often gets the diagnosis *name* right and the code *number* wrong (6 of 50 rows have an exactly matching description with a different code, e.g. "ENDOMETRIOSIS, UNSPECIFIED" returned as N83 where the gold is N809). It learned the format and the names much better than the numbers. This agrees with the training-time result (`icd_match` 0/20). The base model scatters 3 to 6 codes per note.

### 3. Hallucination: the adapter inherits the dataset's habits

Two methods, both comparing each note to its transcript: deterministic rules (numbers, drug+dose pairs, "Dr. Name" must appear in the transcript) and an LLM judge (`gpt-4.1-mini`, temperature 0, blind to which model wrote the note). The gold notes are scored too, because they are themselves elaborated beyond the transcripts (27% of their numbers never appear in the transcript).

| | gold (control) | base | fine-tuned |
|---|---|---|---|
| Rules: share of numbers not in the transcript | 16.7% | 0% | 12.8% |
| Judge: unsupported claims per note | 3.34 | 0.20 | 3.48 |
| Judge: notes with no unsupported claim | 2 | 41 | 2 |
| Judge: notes with at least one *major* claim | 7 | 4 | 9 |
| Judge: key transcript facts missing per note | 5.0 | 6.7 | 5.1 |

- **The adapter fabricates at the gold rate, not below it.** The largest category is invented demographics: 49 of 50 fine-tuned notes and 48 of 50 gold notes state an age, ethnicity or name that is not in the transcript (typically a "45-year-old" patient). The model learned this from the data.
- **The base model is cleaner but thinner.** It almost never invents anything, but it also omits more key facts (6.7 vs. 5.1), so part of its low hallucination rate comes from saying less.
- **Major (clinically misleading) claims still occur in the adapter's notes**, in 9 of 50. Examples: "hypertension currently managed with Metformin", referrals to doctors who are never mentioned, and "denies jaundice" where the transcript says otherwise. All 17 major flags (both models) were read by hand and were genuine.

### 4. Latency: slower, partly by design

| | base | fine-tuned |
|---|---|---|
| Median / mean request (s) | 14.1 / 18.0 | 51.2 / 57.0 |
| p90 / max (s) | 17.6 / 199.3 | 64.9 / 316.9 |
| Median output (characters) | 1,552 | 3,061 |
| Median seconds per 1,000 characters | 8.9 | 16.7 |

The 3.6x gap at the median splits into roughly 2x from longer outputs (the adapter writes fuller notes) and roughly 1.9x from slower generation per character, which is consistent with serving an unmerged LoRA adapter on a T4. Merging removes that overhead (section 5). The maxima are probably cold-start events, so prefer the medians. Characters stand in for tokens because token counts were not recorded.

### 5. Merging the adapter into the base weights

The adapter was also merged into the fp16 base model (`serving/modal_merge.py`) and the same 50 rows were run through it on the same T4 setup, one request at a time. The merge was checked before it was saved: three weights recomputed by hand matched exactly in float32, and the merged and adapter-applied models' logits differed by at most 0.000045 with identical top tokens.

| | base | fine-tuned (adapter) | fine-tuned (merged) |
|---|---|---|---|
| Median / mean request (s) | 14.1 / 18.0 | 51.2 / 57.0 | **25.2 / 28.7** |
| p90 / max (s) | 17.6 / 199.3 | 64.9 / 316.9 | 33.0 / 182.0 |
| Median seconds per 1,000 characters | 8.9 | 16.7 | **8.2** |
| Strict valid | 0/49 | 50/50 | 49/50 |
| Code category hit | 0.06 | 0.24 | 0.24 |
| Exact code after gateway repair | n/a | 11/50 | 11/50 |
| Code description hit | 0.06 | 0.12 | 0.12 |
| Judge: unsupported claims per note | 0.20 | 3.48 | 3.70 |
| Judge: notes with a major claim | 4 | 9 | 9 |
| Judge: missing key facts per note | 6.7 | 5.06 | 5.34 |

- **Speed:** merging removes the adapter's per-token cost. The merged model is 2.0x faster than the unmerged adapter at the median, and per character it is as fast as the plain base model (8.2s vs 8.9s per 1,000 characters). It writes the same amount (median 3,056 vs 3,061 characters).
- **Quality:** equivalent within noise on codes, the judge's major-claim count and the rule-based checks. The judge's means are slightly worse (3.70 vs 3.48 unsupported claims), which at n = 50 is not distinguishable from the run-to-run variation described below.
- **The outputs are not identical.** Only 18 of 50 are byte-identical to the unmerged adapter's, the suggested codes agree on 45 of 50, and the mean text similarity is 0.88. At `temperature` 0 a rounding difference in the fp16 weights can flip one token early, after which a long generation takes a different path. This is expected given the tiny numeric difference measured above, not a sign of a faulty merge, but it means the merged model is a *different* model to evaluate, not a transparent substitute.
- **One real regression: 49/50 strict-valid.** On row 28 the merged model returned valid JSON with `assessment` as a nested object instead of a string (5,810 characters vs 3,450 for the adapter). The gateway's retry layer would re-ask on a failure like this; the replay in `gateway_replay_merged.json` does not retry, so it counts the row as unusable. With one failure in 50 this cannot be told apart from chance, so watch the gateway's `attempts` field.

## What to do about it

| Finding | Where to fix it |
|---|---|
| Wrong ICD code numbers, right names | **Gateway (Pillar 5):** validate each code against an ICD-10 table and check it matches the description; repair by description lookup, or retry. Measure the gain with `rescore.py`. |
| Invented demographics and plan items | **Gateway:** flag any age, sex, name or referral in a note that the transcript does not state. **Training data:** strip invented demographics from the gold notes before the next training round. |
| Major claims (wrong drug-condition pairings, contradictions) | **Gateway / sign-off (Pillars 5 and 6):** treat notes with unsupported medications or contradictions as needing human review. |
| Latency | Faster GPU class; record token counts in future runs. Merging the adapter removes the overhead (section 5, now the default). Still to do: a faster GPU class, and token counts in future runs. |
| Format | Nothing to fix. Keep the strict parser; do not add fence-stripping for the adapter. |

## Limitations

- 50 test rows and a single run per model; no confidence intervals.
- The gold labels are synthetic and elaborated, so "gold" is a reference, not clinical ground truth.
- One judge model, one pass, no human labels beyond the 17 major flags. Treat the judge's missing-fact count as a relative measure (it ignored its own cap of 8 on some notes). The judge did not assess ICD codes.
- Rule-based checks only catch numbers, drug+dose pairs and doctor names; they miss invented findings and contradictions.
- Latency used one T4 and sequential requests, and includes cold-start outliers.
- The merged run is a single run of a model whose outputs differ from the adapter's (see section 5), so the 49/50 vs 50/50 format result and the small judge differences are within what one more run could change.
- Configs B and C were never trained, so there is no comparison between adapter configurations.

## Reproducing

See [README.md](README.md) for the commands. Raw outputs: `results_base.json`, `results_finetuned.json`, `rescored.json`, `hallucination_rules.json`, `hallucination_judge.json`.
