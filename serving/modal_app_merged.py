"""Serves the merged model (base + config A adapter, already combined) on Modal: plain vLLM,
no LoRA flags, no per-token adapter overhead. Fully separate from modal_app.py (the unmerged
adapter) and modal_app_base.py (the plain base model): its own app name and URL.

Needs the Volume written by serving/modal_merge.py.

Usage:
    python -m modal deploy serving/modal_app_merged.py
"""
import pathlib
import subprocess

import modal

HERE = pathlib.Path(__file__).parent  # serving/

image = modal.Image.from_dockerfile(
    str(HERE / "Dockerfile"), context_dir=HERE, add_python="3.12"
).entrypoint([])  # see modal_app.py for why this is needed

app = modal.App("cliniflow-vllm-merged", image=image)
volume = modal.Volume.from_name("cliniflow-merged")


@app.function(
    gpu="T4",  # same GPU class as the adapter and base apps, so latency is comparable
    timeout=60 * 30,
    scaledown_window=60 * 10,
    max_containers=1,
    volumes={"/merged": volume},
)
@modal.concurrent(max_inputs=10)  # same as modal_app.py: lets vLLM's batching see concurrent requests
# Callers must send the Modal-Key / Modal-Secret headers of a proxy-auth token (serving/README.md),
# so the GPU endpoint is not open to anyone who learns the URL.
@modal.web_server(port=8000, startup_timeout=600, requires_proxy_auth=True)
def serve_merged():
    # Same flags as modal_app_base.py: the only difference is the weights.
    cmd = (
        "PATH=/usr/bin:/bin:$PATH python3 -m vllm.entrypoints.openai.api_server "
        "--model /merged/config_A_merged "
        "--served-model-name clinical-json-extractor-merged "
        "--dtype half --max-model-len 4096 "
        "--host 0.0.0.0 --port 8000"
    )
    subprocess.Popen(cmd, shell=True, cwd="/serving")
