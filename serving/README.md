# Serving with vLLM (Pillar 3)

Status: the server, client, and load test are proven working end-to-end on a Colab T4. Nothing is deployed on AWS yet &mdash; blocked on GPU quota approval, and `launch_ec2.py`/`stop_ec2.py` have not been run against real AWS yet (written and syntax-checked only).

This is the learning guide for the serving layer. It explains what vLLM is, why CliniFlow uses it, and the steps to run it on AWS, in the order we'll do them.

## 1. What vLLM is, in plain terms

A language model is a large file of numbers. Running it, called **inference**, means feeding it text and getting text back. That's slow and memory-hungry, and doing it naively wastes most of the GPU.

vLLM is an open-source program that runs a model efficiently behind an HTTP API. It does three things that matter for us:

- **Keeps the GPU busy.** Requests arrive at different times. vLLM adds new requests to the batch that's already running, instead of waiting for the batch to finish. This is called *continuous batching*.
- **Uses memory carefully.** While generating, the model stores intermediate state (the KV cache) for every token. vLLM manages that cache in fixed-size blocks, like pages of memory, so less is wasted. This is called *PagedAttention*.
- **Speaks the OpenAI API format.** Any client written for OpenAI's chat endpoint can call it by changing the URL. Our FastAPI gateway in Pillar 5 will do exactly that.

## 2. Why CliniFlow uses it, and what it replaces

Our model is the base Llama 3.2 3B plus a small LoRA adapter. Two ways to serve that:

| Approach | What it means | Problem |
|---|---|---|
| Merge the adapter into the base model, then serve | One full model file per adapter version | Re-upload several GB per change; one model per adapter |
| **vLLM with LoRA support (chosen)** | One base model loaded once; adapters attached per request | Needs a GPU server, but one server can serve several adapters |

The design doc chose the second option. Adapters stay small and separate, and the base model loads once.

## 3. The core idea: base once, adapter per request

```
            one running vLLM server
  ┌──────────────────────────────────────────┐
  │  base model (Llama 3.2 3B, fp16)  ~6 GB  │  loaded once
  │  ┌─────────────────────────────────────┐ │
  │  │ adapter: clinical-json-extractor    │ │  small, swapped per request
  │  └─────────────────────────────────────┘ │
  └──────────────────────────────────────────┘
        ▲                         │
   request: model="clinical-      response: SOAP JSON
   json-extractor", transcript    
```

The `model` field in the request selects the adapter. A future adapter, such as `patient-summarizer` from the design doc, could be added to the same server without a second GPU.

## 4. Hardware, and why this instance

- **Instance:** g4dn.xlarge, EC2 Spot, eu-north-1.
- **GPU:** one NVIDIA T4 with 16 GB of memory.
- **Why it fits:** the 3B model in fp16 is about 6 GB of weights. The rest of the memory holds the KV cache and the adapter. This is the sizing reasoning from the design doc.
- **Precision:** the T4 uses fp16, not bf16. vLLM is started with `--dtype half` for that reason.
- **Cost:** check the live EC2 price for eu-north-1 before launching. Spot is much cheaper than on-demand, but it can be interrupted with a 2-minute warning. Stop the instance whenever you're not using it.

## 5. The steps, in order

**Step 1. Get GPU quota.** Covered in the request steps above. Nothing else can launch until this is approved. Check status in Service Quotas &rarr; Quota request history.

**Step 2. One-time IAM setup (console, by hand).** `cliniflow-dev` doesn't have EC2 permissions by default, and an EC2 instance needs its own identity to read S3 — handing it your personal access key would be a bigger blast radius than it needs. Two things to create, each a deliberate human step (not scripted), so a least-privilege identity is never self-granted:

1. **Attach the EC2 policy to `cliniflow-dev`.** IAM &rarr; Users &rarr; `cliniflow-dev` &rarr; Add permissions &rarr; Create inline policy &rarr; JSON &rarr; paste `registry/iam/cliniflow-dev-ec2-policy.json` (fill in `<ACCOUNT_ID>`) &rarr; name it `cliniflow-pillar3-ec2`.
2. **Create the instance role.** IAM &rarr; Roles &rarr; Create role &rarr; trusted entity **EC2** &rarr; no policies yet &rarr; name it `cliniflow-ec2-serve` &rarr; create. Then open it &rarr; Add permissions &rarr; Create inline policy &rarr; JSON &rarr; paste `registry/iam/cliniflow-ec2-serve-policy.json` &rarr; name it `cliniflow-s3-read`. This role is read-only on the adapter prefix &mdash; narrower than `cliniflow-dev`, since the instance only ever needs to download, never upload.

**Step 3. Find an AMI (Amazon Machine Image).** This is the pre-built OS image the instance boots from &mdash; we want one with the NVIDIA driver and Docker already installed, so we don't install them by hand. In the EC2 console &rarr; AMI Catalog &rarr; search **"Deep Learning"** &rarr; filter to a Ubuntu-based AWS Deep Learning AMI with NVIDIA driver + Docker &rarr; confirm the region is **eu-north-1** &rarr; copy its AMI ID (`ami-...`). AMI IDs are region-specific and change over time, which is why this is a lookup step rather than a hardcoded value in `launch_ec2.py`.

**Step 4. Launch the instance.**

```bash
export EC2_AMI_ID=ami-xxxxxxxxxxxxxxxxx   # from Step 3
python serving/launch_ec2.py
```

`serving/launch_ec2.py` is new: it finds your default VPC/subnet, creates a security group open **only to your current public IP** on ports 22 (SSH) and 8000 (the API), creates an SSH key pair (saved to `~/.ssh/cliniflow-ec2-key.pem`, outside the repo and already gitignored), and requests one `g4dn.xlarge` Spot instance tagged `cliniflow-vllm` with the `cliniflow-ec2-serve` role attached. It prints the public IP and the exact SSH command when done. Re-running it is safe: it reuses the instance if one is already running instead of launching a second.

This script has not been run yet &mdash; it's written and syntax-checked, but the first real run is also its first real test, since quota approval was still pending.

**Step 5. Get the model and the adapter onto the instance.** SSH in with the command `launch_ec2.py` printed, then:
- The base model: the full-precision Llama 3.2 3B Instruct from Hugging Face. The adapter was trained on the 4-bit Unsloth build, which is a different checkpoint. LoRA adapters generally work across these builds, but we'll confirm it by testing rather than assume it.
- The adapter: `aws s3 sync s3://cliniflow-adapters-paul-2026/adapters/config_A/v1/ ./adapter/`. The instance's `cliniflow-ec2-serve` role already allows this read, with no key copying needed. vLLM loads adapters from local files, so the instance needs a copy first.

**Step 6. Start the server.** The command, with each flag explained:

```bash
vllm serve <base-model-repo> \
  --dtype half \                              # T4 uses fp16, not bf16
  --max-model-len 4096 \                      # longest input+output in tokens; matches training
  --enable-lora \                             # turn on adapter support
  --max-lora-rank 16 \                        # must be >= the adapter's rank (16)
  --lora-modules clinical-json-extractor=./adapter \   # name the adapter for requests
  --gpu-memory-utilization 0.90               # how much GPU memory vLLM may claim
```

There's no server-side setting for the system prompt. The client sends it with each request (see Step 7). Check flag names against the vLLM version we actually install before running this.

**Step 7. Send a request.** The OpenAI-style call:

```bash
curl http://<instance-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clinical-json-extractor",
    "messages": [
      {"role": "system", "content": "<contents of system_prompt.txt>"},
      {"role": "user", "content": "<transcript>"}
    ],
    "max_tokens": 1500,
    "temperature": 0
  }'
```

Two details matter:
- **The system prompt must match training exactly.** A changed character can change the output. That's why it's saved with the adapter.
- **`temperature: 0`** makes output deterministic, which makes testing comparable across runs.

**Step 8. Check the output.** Run the same validator we used in Pillar 2 on the response. Is it valid JSON, and does it pass the schema?

**Step 9. Load test.** Send 5–20 requests at once and record:
- throughput (tokens per second)
- time to first token (TTFT), which is how long the user waits before anything appears

These are the numbers the design doc asks us to document as evidence of inference optimization.

**Step 10. Protect the instance, then stop it.** Watch for the spot interruption notice (a 2-minute warning if AWS reclaims the capacity). When you're done demoing:

```bash
python serving/stop_ec2.py              # stop, can restart later
python serving/stop_ec2.py --terminate  # delete it entirely
```

`stop_ec2.py` is new, alongside `launch_ec2.py`. Both find the instance by its `cliniflow-vllm` tag, so neither needs an instance ID typed in by hand.

## 6. Gotchas to remember

- `--max-lora-rank` must be at least the adapter's rank. Ours is 16.
- The system prompt must be identical at training and serving.
- The T4 needs `--dtype half`. bf16 won't run.
- If the server runs out of memory, lower `--max-model-len` or `--gpu-memory-utilization`.
- The base model in Hugging Face and the adapter's training base must be compatible. We'll check this in Step 6 before trusting results.

## 7. Decisions already made

- **Base repo:** `unsloth/Llama-3.2-3B-Instruct`, assumed but not yet confirmed accessible &mdash; verify on Hugging Face before Step 6.
- **Colab rehearsal:** done. The full server, one real request, and a load test all ran successfully against the trained adapter on a free T4 before any AWS spend.
- **Load test location:** runs from your laptop (via `serving/load_test.py`), not from the instance itself, so no GPU time is spent on the harness itself, only on serving the requests it sends.
