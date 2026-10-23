# Evaluation glossary and guide

A plain-language guide to what the Pillar 4 evals measure, with real examples from our test set. Read this before `BENCHMARK_REPORT.md` if the terms are new.

## The core idea: marking an exam

- **Test set**: 50 doctor-patient transcripts the model never saw in training (`training/colab/test.jsonl`).
- **Gold**: the dataset's own reference note and ICD code for each transcript. The answer key.
- **Base model**: the untrained student. **Fine-tuned model**: the student who studied our training data.
- **Evaluation**: give both the same paper, mark the answers, report where each did better.

A note can be good at one thing and bad at another, so we mark four separate things.

## 1. Format: did it follow the template?

We asked for JSON with four text fields (subjective, objective, assessment, plan) and a list of suggested codes. "Valid" means a program can read it and every required part is present.

- **Strict valid**: the reply is only JSON. The base model wrapped its JSON in a markdown code fence (three backticks and the word `json`), which breaks a program reading raw text. Strictly, the base model scores 0/50.
- **Lenient valid**: the same check after stripping the fence. The base model scores 48/49.

The difference separates "cannot produce the structure" (the base model can) from "adds wrapping a program cannot read" (it does). The fine-tuned model passes both, 50/50.

## 2. Code accuracy: did it pick the right ICD-10 code?

ICD-10 is a catalogue of diagnosis codes. We compare the predicted code with the gold code.

- **Exact**: identical. Neither model ever managed this.
- **Category hit**: the first 3 characters match. Row 7: gold `E1122` (type 2 diabetes with kidney disease), fine-tuned `E1130` (also type 2 diabetes, a different complication). Both start with `E11`: right family, wrong detail.
- **Description hit**: the diagnosis name matches. Row 2: gold `N809` "ENDOMETRIOSIS, UNSPECIFIED"; the model wrote `N83`, also labelled "ENDOMETRIOSIS, UNSPECIFIED". Right name, wrong number.

The model learned diagnosis names much better than their numbers.

## 3. Hallucination: did it make things up?

A **hallucination** is a claim in the note that the transcript does not support. Example (row 49): the note says hypertension is "managed with Metformin", but Metformin is a diabetes drug.

Two ways we measure it:

- **Rule-based checks**: cheap and precise but narrow. Every number, drug-and-dose pair and "Dr. Name" in the note must appear in the transcript. They cannot spot an invented finding or a contradiction.
- **LLM-as-judge**: another AI reads the transcript and the note and lists unsupported claims. Each is tagged **minor** (e.g. an invented patient age) or **major** (a wrong drug, dose, diagnosis or contradiction a clinician could be misled by). It sees one note at a time and does not know which model wrote it.

Related terms:

- **Control / baseline**: we judge the gold notes too, because they also invent details (27% of their numbers are not in the transcript). The question is not "is it zero?" but "is it better or worse than the gold standard?". The fine-tuned model hallucinates at about the gold rate: 3.48 unsupported claims per note against 3.34.
- **Missing facts**: a note can look clean by saying very little. The base model had 0.2 unsupported claims per note but missed 6.7 key facts, against 5.1 for the fine-tuned model.
- **Why no single "hallucination rate"**: we report unsupported claims per note, notes with at least one major claim, and missing facts per note. One number would hide the trade-off between saying little and making things up.

## 4. Speed

- **Latency**: time per request.
- **Median**: the middle value. Better than the **mean** (average) here, because a few cold-start requests of 200 to 300 seconds distort the mean.
- **p90**: the time that 90% of requests beat.
- **Seconds per 1,000 characters**: adjusts for the fine-tuned model writing longer notes.

## Vocabulary

| Term | Meaning |
|---|---|
| **Assertion** | One automatic check on one reply that returns pass or fail, such as "is it bare JSON?" |
| **Rubric** | Written grading instructions for an LLM judge, the marking scheme |
| **Grader / judge** | The AI model that applies a rubric |
| **Deterministic** | Gives the same answer every time (format checks, code matching) |
| **LLM-judged** | Depends on an AI's opinion, so it can vary and make mistakes |
| **Provider** | In Promptfoo, a model under test |
| **Replay / live** | Replay re-scores saved replies (free). Live sends fresh requests to the servers (costs GPU time) |
| **temperature 0** | The model always picks its most likely word, so reruns are comparable |
| **n = 50** | Sample size. A difference of 2 or 3 rows could be luck |
| **Recall** | Of the real problems, the share the checker caught |
| **Precision** | Of the things the checker flagged, the share that were real |
| **Over-flagging** | High recall, low precision. `gpt-5-mini` caught 6 of our 9 known fine-tuned problems but flagged 21 notes in total |

## With and without Promptfoo

| Question | Python scripts | Promptfoo |
|---|---|---|
| Send transcripts to a model | `run_eval.py` | its provider |
| Format and code checks | `run_eval.py`, `rescore.py` | assertions in the config |
| Hallucination | `hallucination_rules.py`, `hallucination_judge.py` | a rubric assertion (one yes/no call per note) |
| Results | JSON files and a printed table | a browsable grid (`promptfoo view`) |

Both measure the same things and the deterministic checks agree exactly, which is how we know both implementations are correct. The rubric is the one place they differ: one yes/no call per note is much less sensitive than the standalone judge's claim-by-claim list. Promptfoo's main value is convenience, and it pays off most when the merged model is added as a third column.

## How the results add up

1. Fine-tuning fixed the format: 50/50 strict-valid against 0/50.
2. Code numbers are still wrong (0 exact, 24% category), so the gateway must validate and repair them.
3. The model copies the dataset's habit of inventing details, at about the gold rate, and some are dangerous (9 of 50 notes have a major claim). That points to the training data and to human sign-off in Pillar 6.
4. It is about 3.6x slower, which motivates the merged-adapter experiment in `docs/BACKLOG.md`.
