"""The CliniFlow model: the fine-tuned SOAP-note model, served with vLLM on Modal.

This is the one deployment the gateway, the agent tools and the evals use. It serves the fine-tuned
adapter already merged into the base weights (built by serving/modal_merge.py), so vLLM runs a plain
model with no per-token adapter overhead. The comparison deployments (the plain base model and the
adapter served unmerged) live in serving/experiments/ and exist only to reproduce the benchmark.

Needs the Volume written by serving/modal_merge.py.

Usage:
    python -m modal deploy serving/modal_app.py
"""
import os
import pathlib
import subprocess

import modal

HERE = pathlib.Path(__file__).parent  # serving/
# Optional: deploy with CLINIFLOW_PROXY_AUTH=1 to require a Modal proxy-auth token (see serving/README.md).
PROXY_AUTH = os.environ.get("CLINIFLOW_PROXY_AUTH") == "1"

image = modal.Image.from_dockerfile(
    str(HERE / "Dockerfile"), context_dir=HERE, add_python="3.12"
).entrypoint([])  # the Dockerfile's own ENTRYPOINT is for `docker run`; clear it so Modal controls startup

app = modal.App("cliniflow-soap", image=image)
volume = modal.Volume.from_name("cliniflow-merged")


@app.function(
    gpu="T4",  # the same GPU class as the comparison deployments, so latencies are comparable
    timeout=60 * 30,
    scaledown_window=60 * 10,
    max_containers=1,
    volumes={"/merged": volume},
)
@modal.concurrent(max_inputs=10)  # without this Modal forwards one request at a time and vLLM cannot batch
@modal.web_server(port=8000, startup_timeout=600, requires_proxy_auth=PROXY_AUTH)
def serve():
    cmd = (
        "PATH=/usr/bin:/bin:$PATH python3 -m vllm.entrypoints.openai.api_server "
        "--model /merged/config_A_merged "
        "--served-model-name cliniflow-soap "
        "--dtype half --max-model-len 4096 "
        "--host 0.0.0.0 --port 8000"
    )
    subprocess.Popen(cmd, shell=True, cwd="/serving")
