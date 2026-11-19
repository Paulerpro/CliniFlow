"""Deploys the CliniFlow vLLM server on Modal's on-demand GPUs, reusing
serving/Dockerfile unchanged. The image is portable by design: the same
Dockerfile also runs under plain `docker run --gpus all` on any other GPU host.

Usage:
    python -m modal deploy serving/experiments/modal_app_adapter.py
"""
import os
import pathlib
import subprocess

import modal

HERE = pathlib.Path(__file__).resolve().parent.parent  # serving/ (this file lives in serving/experiments/)
PROXY_AUTH = os.environ.get("CLINIFLOW_PROXY_AUTH") == "1"
ADAPTER_DIR_LOCAL = HERE.parent / "registry" / "adapters" / "config_A"

image = (
    modal.Image.from_dockerfile(str(HERE / "Dockerfile"), context_dir=HERE, add_python="3.12")
    # The Dockerfile's own ENTRYPOINT (correct for `docker run`) would otherwise auto-launch
    # start_server.sh the instant the container starts, before this module's serve() ever
    # runs - bypassing the adapter/env setup below entirely. Clear it so Modal's function
    # invocation is what actually controls startup here.
    .entrypoint([])
    .add_local_dir(
        str(ADAPTER_DIR_LOCAL),
        remote_path="/adapter",
        ignore=[".cache", ".gitattributes", ".git"],
    )
)

app = modal.App("cliniflow-vllm", image=image)


@app.function(
    gpu="T4",  # same GPU class already proven to work, on Colab
    timeout=60 * 30,       # 30 min: a rehearsal/demo session, not an always-on service
    scaledown_window=60 * 10,  # shut the container down 10 min after the last request
    max_containers=1,  # single demo instance; avoids accumulating GPU allocations
)
# Without this, Modal forwards only 1 request at a time to the container and queues
# the rest externally at its own gateway - vLLM's continuous batching never sees more
# than one request, no matter the client's concurrency. 10 is within the measured KV
# cache headroom (~11x concurrent full-length requests at max_model_len=4096).
@modal.concurrent(max_inputs=10)
# Optional: deploy with CLINIFLOW_PROXY_AUTH=1 to require a Modal proxy-auth token (see serving/README.md).
@modal.web_server(port=8000, startup_timeout=600, requires_proxy_auth=PROXY_AUTH)
def serve():
    # PATH prepended so `python3` resolves to the base image's interpreter (has vllm installed),
    # not the one `add_python` injected at /usr/local/bin (needed only so Modal can detect a
    # Python version for its own runtime, otherwise unused).
    cmd = (
        "PATH=/usr/bin:/bin:$PATH "
        "ADAPTER_DIR=/adapter BASE_MODEL=unsloth/Llama-3.2-3B-Instruct bash start_server.sh"
    )
    # non-blocking: the web_server decorator waits for port 8000 to answer, then proxies to it
    subprocess.Popen(cmd, shell=True, cwd="/serving")
