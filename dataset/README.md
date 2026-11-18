# Dataset

Training and evaluation data for CliniFlow, derived from **MedSynth** (Mianroodi et al., [arXiv:2508.01401](https://arxiv.org/abs/2508.01401), [Hugging Face](https://huggingface.co/datasets/Ahmad0067/MedSynth)): synthetic doctor-patient dialogues paired with SOAP notes and ICD-10 labels, all written by GPT-4o. No real patient data.

**The data files are not in this repository.** MedSynth's dataset card states no license, so no right to redistribute it is granted here (see [`NOTICE`](../NOTICE)). The scripts below regenerate the files exactly, and `data_manifest.json` lets you verify that.

## Regenerate

```bash
pip install datasets pydantic
cd dataset
python dataset_gen.py          # downloads MedSynth (10,240 rows)        -> medsynth_soap.jsonl
python raw/1K.py               # seeded 1,000-row sample                 -> medsynth_1k_soap.jsonl
python prep_and_validate.py    # parse SOAP sections, validate schema     -> medsynth_1k_schema_valid.jsonl
python inject_noise.py         # seeded noise on the dialogue only        -> medsynth_1k_noisy.jsonl
python split_dataset.py        # 50 test / 49 val / 901 train             -> ../training/colab/{test,val,train}.jsonl
cd ..
python dataset/verify_data.py  # compare against data_manifest.json
```

Every random step is seeded (the sample, the noise per example, the split), and `test.jsonl` is the held-out set behind every number in `evals/`. Never train on it.

**Verified:** on 2026-11-18 the pipeline was rerun from a fresh download in a clean folder and compared with the files the experiments used. The raw download, the 1K sample, the noisy file and the train, validation and test splits are byte-identical; the schema-valid file is identical once parsed (see the repair below). The check hashes content with line endings normalised, so a run on Linux or macOS matches a run on Windows. It does assume the upstream dataset has not changed since.

## The pipeline

| Step | What it does |
|---|---|
| `dataset_gen.py` | Downloads MedSynth |
| `raw/1K.py` | Draws a seeded random sample of 1,000 rows |
| `prep_and_validate.py` | Cleans the dialogue, parses the markdown note into subjective / objective / assessment / plan, builds the target JSON with the ICD-10 label, validates it against the schema, and writes rejects to `medsynth_1k_dropped.jsonl` |
| `inject_noise.py` | Makes the input messier (20% clean, 40% light, 40% heavy). Rule-based and input-only; numbers are asserted unchanged. It is not real speech-recognition noise |
| `split_dataset.py` | Seeded shuffle, then 50 test, 49 validation, the rest train |

## Things to know about this data

- **One row was repaired by hand, and the repair is now in the script.** Source line 309 (a COPD note) has an empty Objective section, so validation rejects it. The data used for training kept that row, with the objective written as "Not documented — encounter conducted via telemedicine; no in-person vitals or physical exam performed.", appended as the last row. That edit was not in any script, so the scripts alone produced 999 rows, shifted the shuffle and changed the test set (only 8 of the 50 test notes matched). It is now recorded as `MANUAL_REPAIRS` in `prep_and_validate.py`. The same row also carries the source's own label for that note, an HPV-screening code (`Z1151`), which does not fit a COPD note, so its confidence is set to `low`.
- **Notes were written first, dialogues generated from them.** MedSynth builds the note and then a dialogue "that led to that note". Training dialogue to note therefore asks the model to recover details the conversation never states (age, ethnicity, names, vitals, referrals). Judging all 901 training notes against their conversations found 3.37 unsupported claims per note, 19% of notes with a major one, and only 5.5% fully supported. This is the root of the hallucination finding in [`evals/BENCHMARK_REPORT.md`](../evals/BENCHMARK_REPORT.md).
- **ICD-10 labels come from MedSynth and are not independently checked,** so some are wrong (see the repaired row above).
- The dialogue is the model's input and the note is its target; the system prompt tells the model to include only facts stated in the conversation, which the targets often do not.
