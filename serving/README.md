# Serving with vLLM (Pillar 3)

Serves the CliniFlow model, the fine-tuned SOAP-note model, behind an OpenAI-compatible HTTP API with vLLM. There is one deployment that everything else (the gateway, the agent tools, the evals) talks to: `serving/modal_app.py`, which serves the fine-tuned adapter **merged into the base weights** under the model name `cliniflow-soap`.

The plain base model and the adapter served unmerged also exist, in [`experiments/`](experiments), but only so the benchmark can be reproduced. They are not alternative ways to use the project.

## What vLLM does

vLLM runs a language model efficiently behind an HTTP API, built around two techniques:

- **Continuous batching.** New requests join the batch already running on the GPU instead of waiting for it to finish, keeping the GPU busy.
- **PagedAttention.** The per-token generation state (the KV cache) is managed in fixed-size blocks rather than allocated as one contiguous reservation, reducing wasted GPU memory.

It exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so any existing OpenAI-style client can point at it unchanged.

## Deploy the CliniFlow model (Modal)

```bash
pip install modal
python -m modal setup

python -m modal run serving/modal_merge.py     # once, and again after each retrain: builds the merged weights
python -m modal deploy serving/modal_app.py    # serves them; prints the URL
```

- **`modal_merge.py`** merges the adapter in `registry/adapters/config_A` into the fp16 base model on Modal (it needs about 13 GB of RAM, so not on a laptop) and writes the result, 6.4 GB, to the Volume `cliniflow-merged`. It recomputes three weights by hand and compares the merged and adapter-applied logits, and writes nothing if either check fails (last run: maximum logit difference 0.000045, identical top tokens). A `merge_manifest.json` beside the weights records the base model, the adapter's SHA-256, the LoRA scale and the library versions.
- **`modal_app.py`** builds `serving/Dockerfile` as the container image, mounts that Volume, asks for a T4 GPU on demand and exposes the vLLM server through Modal's `web_server`. The app is `cliniflow-soap`, and the model name clients send is `cliniflow-soap`. Two details specific to building from a custom Dockerfile on Modal:
  - The Dockerfile's own `ENTRYPOINT` is cleared (`.entrypoint([])`) so Modal's function invocation controls startup, not the image auto-launching the server.
  - `PATH` is set explicitly in the launch command so `python3` resolves to the base image's interpreter (the one with vLLM), not the separate one `add_python` provisions for Modal's runtime.
- A cold container takes several minutes to answer its first request; after that it stays warm for 10 minutes.

### Why the adapter is merged

Training and the registry deal in the small adapter (a few MB, the source of truth), but serving uses a merged copy. The project first served the adapter unmerged, expecting to add more adapters later. With a single adapter that flexibility went unused while the adapter was applied on every generated token. Measured on the same 50 test rows, T4, `temperature` 0: median request **51.2 s unmerged vs 25.2 s merged** (2.0x), and 16.7 s vs 8.2 s per 1,000 output characters, about the plain base model's speed. Quality is equivalent within noise; see [`../evals/BENCHMARK_REPORT.md`](../evals/BENCHMARK_REPORT.md), including the one format failure the merged model had.

### Optional: require a token on the endpoint

By default the deployed URL accepts any request. To require a Modal proxy-auth token, deploy with `CLINIFLOW_PROXY_AUTH=1`; Modal then answers HTTP 401 to requests without a valid token.

1. Create a token in the Modal dashboard (Settings, Proxy Auth Tokens) and put it in `.env` (gitignored): `MODAL_PROXY_TOKEN_ID=wk-...` and `MODAL_PROXY_TOKEN_SECRET=ws-...`. These are not your Modal CLI credentials (`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`).
2. Deploy with the variable set: `CLINIFLOW_PROXY_AUTH=1 python -m modal deploy serving/modal_app.py`.
3. The clients send the `Modal-Key` and `Modal-Secret` headers automatically when both variables are set: the gateway (it reads `.env`), `serving/client.py`, `serving/load_test.py` and `evals/run_eval.py` (these three read the process environment, so `export` them first), and the Promptfoo live config. With curl, add `-H "Modal-Key: $MODAL_PROXY_TOKEN_ID" -H "Modal-Secret: $MODAL_PROXY_TOKEN_SECRET"`.

The header handling is covered by tests (`gateway/tests/test_llm.py`, `test_config.py`); an authenticated deployment has not been exercised end to end.

## Experiments: the comparison deployments

`experiments/` holds the two other deployments the benchmark needs. Each has its own app name and URL and cannot affect the product deployment.

| File | App | Serves | Used for |
|---|---|---|---|
| `experiments/modal_app_base.py` | `cliniflow-vllm-base` | the plain base model, as `base-model` | the "before fine-tuning" reference |
| `experiments/modal_app_adapter.py` | `cliniflow-vllm` | the adapter attached at serve time (vLLM multi-LoRA), as `clinical-json-extractor` | the "before merging" reference |

```bash
python -m modal deploy serving/experiments/modal_app_base.py
python -m modal deploy serving/experiments/modal_app_adapter.py
```

How the adapter path works:

```
            one running vLLM server
  ┌──────────────────────────────────────────┐
  │  base model (Llama 3.2 3B, fp16)  ~6 GB  │  loaded once
  │  ┌─────────────────────────────────────┐ │
  │  │ adapter: clinical-json-extractor    │ │  attached per request
  │  └─────────────────────────────────────┘ │
  └──────────────────────────────────────────┘
        ▲                         │
   request: model="clinical-      response: SOAP JSON
   json-extractor", transcript
```

The `model` field selects the adapter, which is how several task-specific adapters could share one GPU. This project has one, which is why it is merged for serving.

## Hardware requirements

- **Minimum:** an NVIDIA GPU with 16 GB VRAM. The 3B model is about 6 GB in fp16; the rest covers the KV cache (and the adapter, if served unmerged).
- **Precision:** `--dtype half` (fp16). Confirmed working on a T4 (fp16 only, no bf16); also valid on Ampere-class cards with bf16 support.
- **Validated on:** a free-tier T4 (Colab) and a T4 on Modal, end to end over the real API (request in, schema-valid SOAP JSON out).

## The server command

The product deployment runs plain vLLM on the merged weights:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /merged/config_A_merged \
  --served-model-name cliniflow-soap \
  --dtype half \                              # fp16; required on T4
  --max-model-len 4096 \                      # matches training's max_seq_length
  --host 0.0.0.0 --port 8000
```

The unmerged adapter server (`start_server.sh`, used by the experiments, the Docker image and the EC2 and Colab paths below) adds `--enable-lora --max-lora-rank 16 --lora-modules clinical-json-extractor=<adapter-dir>`, with `--model` set to the base model.

The system prompt is not a server setting. The client sends it with every request, read from `system_prompt.txt` saved alongside the adapter; it must match training exactly.

## Calling the server

```bash
export SERVER_URL=https://<your-workspace>--cliniflow-soap-serve.us-east.modal.run   # or http://<host>:8000

# one request, validated against the SOAP schema
python serving/client.py --transcript-file path/to/transcript.txt

# concurrent requests; reports TTFT, latency, throughput
python serving/load_test.py --transcripts-file training/colab/val.jsonl --n 20 --concurrency 5
```

Both scripts are host-agnostic: only `SERVER_URL` changes between hosts. They send the model name `cliniflow-soap`; set `MODEL_NAME=clinical-json-extractor` to talk to the unmerged adapter server.

## Other deployment targets (the unmerged adapter server)

The serving layer is also packaged as a single Docker image (`serving/Dockerfile`), which runs the unmerged adapter server (`start_server.sh`) on any GPU host.

### AWS EC2 (alternate)

Fully scripted, gated on AWS GPU instance quota for the account:

```bash
export EC2_AMI_ID=ami-xxxxxxxxxxxxxxxxx   # a Deep Learning AMI (NVIDIA driver + Docker) in the target region
python serving/launch_ec2.py
```

`launch_ec2.py` provisions a `g4dn.xlarge` Spot instance: creates a security group scoped to the caller's IP, an SSH key pair, and attaches the `cliniflow-ec2-serve` instance role (S3 read-only, so the instance never needs a personal access key). `stop_ec2.py` stops or terminates it by tag.

Required IAM setup (one-time, console): attach `registry/iam/cliniflow-dev-ec2-policy.json` to the deploying user, and create the `cliniflow-ec2-serve` role from `registry/iam/cliniflow-ec2-serve-policy.json`.

Once the instance is running, sync the adapter and start the server:

```bash
aws s3 sync s3://<bucket>/adapters/config_A/v1/ ./adapter/
docker build -t cliniflow-serve ./serving
docker run --gpus all -p 8000:8000 -v $(pwd)/adapter:/adapter -e ADAPTER_DIR=/adapter cliniflow-serve
```

### Colab (validation)

`serving/colab_serve_test.ipynb` runs the server directly (no Docker) on a free T4, used to validate the model, adapter and client end to end before any paid deployment.

## Benchmark (Modal, T4, 20 requests, concurrency 5)

Measured on the unmerged adapter deployment, before the model was merged; the concurrency fix applies to every deployment.

**The bug, and the fix.** The first load test measured a wall time of 1231.9s (TTFT median 197.5s, throughput 11.7 tokens/s) despite `concurrency=5` on the client. vLLM's own engine logs showed `Running: 1 reqs, Waiting: 0 reqs` throughout: the engine never received more than one request at a time. Cause: by default, a single Modal container handles one input at a time; everything else queues externally at Modal's gateway, invisible to the server's own batching. Fix: `@modal.concurrent(max_inputs=10)` on the function (in `modal_app.py` and the experiment apps), which lets Modal actually forward concurrent requests into the container so vLLM's continuous batching can do its job.

**Post-fix, two scenarios** (both clean, fully-completed 20/20 runs; a third post-fix run was discarded as unverified after the result queue was cleared mid-run before it finished):

| Metric | Cold start | Warm container |
|---|---|---|
| Wall time, 20 requests | 525.2s | 261.6s |
| TTFT, median | 2.27s | 1.23s |
| TTFT, p95 | 264.12s | 1.88s |
| Full-request latency, median | 67.28s | 59.48s |
| Full-request latency, p95 | 329.95s | 80.22s |
| Aggregate throughput | ~27.1 tokens/s | ~55.2 tokens/s |

Median TTFT lands in the same 1-2.3s range either way, and that is the real, consistent effect of the fix. The large cold-start p95 isn't a regression: with zero active containers beforehand, one or more of the first concurrent requests arrive while the model is still loading, giving those specific requests a very long TTFT while the rest (once warm) go fast. A real user hitting a cold container will see this; a warm one won't. Worth keeping in mind for anything latency-sensitive built on top of this later.

## Gotchas

- The system prompt must be byte-identical between training and serving.
- fp16 (`--dtype half`) is required on T4-class GPUs; bf16 will not run.
- If the server runs out of memory, lower `--max-model-len` or `--gpu-memory-utilization`.
- `--max-lora-rank` (unmerged adapter server only) must be at least the adapter's rank (16 here).
- The adapter was trained against a 4-bit Unsloth build of the base model; serving uses the full-precision release of the same model. LoRA adapters are generally portable across these, verified by testing rather than assumed.
