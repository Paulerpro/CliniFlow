# Serving with vLLM (Pillar 3)

Serves the fine-tuned model behind an OpenAI-compatible HTTP API with vLLM. Two ways of serving the same fine-tune exist side by side: the **merged** model (the default; plain vLLM on weights that already include the adapter) and the **unmerged** adapter (vLLM's native multi-LoRA: the base model loads once and the adapter attaches per request). See the next section for why merged is the default.

## What vLLM does

vLLM runs a language model efficiently behind an HTTP API, built around two techniques:

- **Continuous batching.** New requests join the batch already running on the GPU instead of waiting for it to finish, keeping the GPU busy.
- **PagedAttention.** The per-token generation state (the KV cache) is managed in fixed-size blocks rather than allocated as one contiguous reservation, reducing wasted GPU memory.

It exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so any existing OpenAI-style client can point at it unchanged.

## Merged vs. adapter-at-serve-time

| Approach | Trade-off |
|---|---|
| **Merged (default)** | Plain vLLM on one set of weights. No per-token adapter work. A full 6.4 GB model per adapter version, rebuilt by `modal_merge.py` after each training change |
| Adapter attached at serve time (`modal_app.py`) | One base model loaded once, the adapter is a few MB and swapped per request, several adapters can share one GPU. The adapter is applied on every generated token |

The project first served the adapter unmerged, expecting to add more adapters. With a single adapter that flexibility was unused, so merging was tried. Measured on the same 50 test rows, T4, `temperature` 0: median request 51.2s unmerged vs **25.2s merged** (2.0x), and 16.7s vs 8.2s per 1,000 output characters, about the plain base model's speed (8.9s). Quality is equivalent within noise; see `evals/BENCHMARK_REPORT.md`, including the one format failure the merged model had.

### The deployments

| File | App | What it serves |
|---|---|---|
| `modal_merge.py` | `cliniflow-merge` (a job, not a server) | Builds the merged model into the Volume `cliniflow-merged`; checks it before saving |
| `modal_app_merged.py` | `cliniflow-vllm-merged` | The merged model, as `clinical-json-extractor-merged` |
| `modal_app.py` | `cliniflow-vllm` | The unmerged adapter, as `clinical-json-extractor` (kept for comparison) |
| `modal_app_base.py` | `cliniflow-vllm-base` | The plain base model, as `base-model` |

Rebuild after retraining: `python -m modal run serving/modal_merge.py`, then `python -m modal deploy serving/modal_app_merged.py`. The merge runs on Modal because it needs about 13 GB of RAM; it recomputes three weights by hand and compares the merged and adapter-applied logits, and writes nothing if either check fails (last run: max logit difference 0.000045, identical top tokens).

### How the unmerged path works

```
            one running vLLM server
  ┌──────────────────────────────────────────┐
  │  base model (Llama 3.2 3B, fp16)  ~6 GB  │  loaded once
  │  ┌─────────────────────────────────────┐ │
  │  │ adapter: clinical-json-extractor    │ │  swapped per request
  │  └─────────────────────────────────────┘ │
  └──────────────────────────────────────────┘
        ▲                         │
   request: model="clinical-      response: SOAP JSON
   json-extractor", transcript
```

The `model` field in the request selects the adapter, so a second task-specific adapter (e.g. `patient-summarizer`) could be added to the same server without a second GPU.

## Hardware requirements

- **Minimum:** an NVIDIA GPU with 16GB VRAM. The 3B model is ~6GB in fp16; the rest covers the KV cache and the adapter.
- **Precision:** `--dtype half` (fp16). Confirmed working on a T4 (fp16-only, no bf16); also valid on Ampere-class cards with bf16 support.
- **Validated on:** a free-tier T4 (Colab) and a T4 on Modal, both serving the trained adapter end-to-end over the real API (request in, schema-valid SOAP JSON out).

## Deployment targets

The serving layer is packaged as a single Docker image (`serving/Dockerfile`), so it's portable across hosts without code changes.

### Modal (primary, live)

```bash
pip install modal
python -m modal setup
python -m modal deploy serving/modal_app.py
```

`serving/modal_app.py` builds `serving/Dockerfile` as the container image, bundles the trained adapter directly into the deployment, requests a T4 GPU on demand, and exposes the vLLM server over Modal's `web_server` networking. Deploy prints the public URL.

Confirmed working end-to-end: a transcript sent to the deployed URL returns a schema-valid SOAP note with a suggested ICD code. Two details specific to building from a custom Dockerfile on Modal, both handled in `modal_app.py`:
- The Dockerfile's own `ENTRYPOINT` is cleared (`.entrypoint([])`) so Modal's function invocation controls container startup instead of the image auto-launching the server on boot.
- `PATH` is set explicitly in the launch command so `python3` resolves to the base image's interpreter (with vLLM installed), not the separate one `add_python` provisions for Modal's own runtime.

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

`serving/colab_serve_test.ipynb` runs the server directly (no Docker) on a free T4, used to validate the model, adapter, and client end-to-end before any paid deployment.

## The server command

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model <base-model-repo> \
  --dtype half \                              # fp16; required on T4
  --max-model-len 4096 \                      # matches training's max_seq_length
  --enable-lora \
  --max-lora-rank 16 \                        # must be >= the adapter's rank (16)
  --lora-modules clinical-json-extractor=<adapter-dir> \
  --host 0.0.0.0 --port 8000
```

The system prompt is not a server setting — it's sent by the client with every request, read from `system_prompt.txt` saved alongside the adapter (it must match training exactly).

## Calling the server

```bash
export SERVER_URL=http://<host>:8000

# one request, validated against the SOAP schema
python serving/client.py --transcript-file path/to/transcript.txt

# concurrent requests; reports TTFT, latency, throughput
python serving/load_test.py --transcripts-file training/colab/val.jsonl --n 20 --concurrency 5
```

Both scripts are host-agnostic — only `SERVER_URL` changes between Colab, Modal, and EC2.

## Benchmark (Modal, T4, 20 requests, concurrency 5)

**The bug, and the fix.** The first load test measured a wall time of 1231.9s (TTFT median 197.5s, throughput 11.7 tokens/s) despite `concurrency=5` on the client. vLLM's own engine logs showed `Running: 1 reqs, Waiting: 0 reqs` throughout — the engine never received more than one request at a time. Cause: by default, a single Modal container handles one input at a time; everything else queues externally at Modal's gateway, invisible to the server's own batching. Fix: `@modal.concurrent(max_inputs=10)` on the function (`serving/modal_app.py`), which lets Modal actually forward concurrent requests into the container so vLLM's continuous batching can do its job.

**Post-fix, two scenarios** (both clean, fully-completed 20/20 runs; a third post-fix run was discarded as unverified after the result queue was cleared mid-run before it finished):

| Metric | Cold start | Warm container |
|---|---|---|
| Wall time, 20 requests | 525.2s | 261.6s |
| TTFT, median | 2.27s | 1.23s |
| TTFT, p95 | 264.12s | 1.88s |
| Full-request latency, median | 67.28s | 59.48s |
| Full-request latency, p95 | 329.95s | 80.22s |
| Aggregate throughput | ~27.1 tokens/s | ~55.2 tokens/s |

Median TTFT lands in the same 1-2.3s range either way — that's the real, consistent effect of the fix. The large cold-start p95 isn't a regression: with zero active containers beforehand, one or more of the first concurrent requests arrive while the model is still loading, giving those specific requests a very long TTFT while the rest (once warm) go fast. A real user hitting a cold container will see this; a warm one won't. Worth keeping in mind for anything latency-sensitive built on top of this later.

## Gotchas

- `--max-lora-rank` must be at least the adapter's rank (16 here).
- The system prompt must be byte-identical between training and serving.
- fp16 (`--dtype half`) is required on T4-class GPUs; bf16 will not run.
- If the server runs out of memory, lower `--max-model-len` or `--gpu-memory-utilization`.
- The adapter was trained against a 4-bit Unsloth build of the base model; serving uses the full-precision release of the same model. LoRA adapters are generally portable across these, verified by testing rather than assumed.
