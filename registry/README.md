# Adapter registry (Pillar 2)

This folder holds the data-validation job and the versioned LoRA adapter registry for CliniFlow. It is the offline side of the system: it prepares and stores artifacts, and it never serves inference.

## What lives where

| Artifact | Location | Notes |
|---|---|---|
| Dataset splits (train / val / test) | `s3://<bucket>/dataset/` | Synthetic only; see the dataset README |
| Validation report | `s3://<bucket>/reports/validation/` | Written by the SageMaker Processing job |
| Adapter v1 (config A) | `s3://<bucket>/adapters/config_A/v1/` | Unmerged LoRA weights, tokenizer, system prompt, manifest |
| Adapter source copy | Hugging Face private repo | Used for reload and sharing |
| IAM policy definitions | `registry/iam/` | Account ID is a placeholder; fill it in when applying |

## Scripts

- `validate_dataset.py`: checks every row against the SOAP target schema (note fields are strings, suggested codes are well-formed). It uses only the Python standard library, so it runs in the stock SageMaker container with no extra installs.
- `launch_processing_job.py`: runs the validator as a SageMaker Processing job on `ml.t3.medium` in eu-north-1, under the `cliniflow-sagemaker-exec` role. It reads `AWS_ACCOUNT_ID` from the environment so the account ID never appears in the repo.

Run locally to check the data before any cloud job:

```
python registry/validate_dataset.py --input-dir training/colab --output-dir registry/report
```

Run in the cloud (requires `aws configure` and `AWS_ACCOUNT_ID` set):

```
python registry/launch_processing_job.py
```

Last cloud run: completed on `ml.t3.medium`, eu-north-1. All 901 train, 49 val, and 50 test rows passed. The report is at `s3://<bucket>/reports/validation/validation_report.json`.

## Why the adapter is stored unmerged

The LoRA adapter is stored as its own small artifact and is never merged into the base model. vLLM loads the base model once and attaches the adapter per request. Merging would mean re-uploading a multi-GB model for every training change and would lock each version to one base checkpoint. Keeping the two separate also lets several adapters share one running server later.

## Why SageMaker stops at data and the registry

SageMaker is used for the data validation job and for storing versioned artifacts. It does not host the model. Serving is handled by vLLM (Pillar 3), so there is one inference path, not two. Creating a SageMaker endpoint or merging weights was deliberately left out of scope.

## Adapter v1 (config A)

Trained with Unsloth 4-bit QLoRA on the 901-row training split, base model `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`.

- LoRA rank 16, alpha 32, learning rate 2e-4, 2 epochs
- Eval loss 0.498, train loss 0.453, about 89 minutes on a Colab T4
- JSON parses: 20/20 on 20 validation examples
- Schema-valid: 20/20
- ICD code matches gold: **0/20**

The model reliably produces well-formed, schema-valid notes. It does not yet predict the correct ICD code, so the code-suggestion output should not be relied on. Config B and C were not run. Improving the code predictions is future work and is documented as a known limitation.

Serving must use the same `system_prompt.txt` that was used in training. vLLM must be started with `--max-lora-rank` of at least 16.

## Known limitations

- The dataset is synthetic. Noise is rule-based, so it does not reproduce real speech-recognition errors.
- The registry holds one adapter version. Multi-version comparison is future work.
- Nothing here is clinically validated or HIPAA-compliant. No real patient data is used or stored.
