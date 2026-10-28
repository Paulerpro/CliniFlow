# Promptfoo setup

Promptfoo runs the same checks as the Python eval scripts and shows the results as a grid (tests by models). New to the terms? See [`../GLOSSARY.md`](../GLOSSARY.md).

| Config | What it does | Cost |
|---|---|---|
| `promptfooconfig.yaml` | **Replay**: re-scores the replies already saved in `evals/results_*.json`. Format and code checks only | Free, no GPU, no API key |
| `promptfooconfig.live.yaml` | **Live**: sends fresh requests to the Modal endpoints. Same checks | GPU time; deploy the apps first |
| `promptfooconfig.hallucination.yaml` | **Replay + LLM rubric** (`gpt-5-mini`, low reasoning effort) for major hallucinations | About $0.35 per run at default effort; run on demand |

## Run

From this folder (Windows: set `PYTHONUTF8=1`):

    npx promptfoo@0.124.1 eval -c promptfooconfig.yaml --no-table -o replay_results.json
    npx promptfoo@0.124.1 view                      # browse the grid

The rubric config needs `OPENAI_API_KEY`: add `--env-file ../../.env`.

## Files

- `tests.js`: builds one test per row of `training/colab/test.jsonl`
- `prompt.json`: system prompt + transcript template
- `replay_provider.js`: returns the saved reply for a row instead of calling a server
- `assertions.js`: strict valid, lenient valid, code-category hit (mirrors `run_eval.py`)

## What to trust

The deterministic checks reproduce the Python numbers exactly (strict valid 0/49 and 50/50, lenient 48/49 and 50/50, category hit 3/49 and 12/50). A cell only shows "pass" when every assertion passes, so the overall pass rate is not comparable to any single metric; read the per-metric breakdown.

The LLM rubric is much less sensitive than the standalone judge (`../hallucination_judge.py`), which lists claims one by one. In tests, `gpt-4.1-mini` with the rubric passed nearly every note. `gpt-5-mini` caught 6 of the 9 fine-tuned notes the standalone judge flagged as major, but flagged 21 notes in total, so it over-flags. Treat the standalone judge as the reference and the rubric as a second opinion.

Run outputs (`replay_*.json`) are gitignored.
