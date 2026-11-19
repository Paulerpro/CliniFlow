"""Deploys the plain base model (no LoRA adapter) on Modal, for the Pillar 4
base-vs-fine-tuned comparison. Fully separate from the product app (serving/modal_app.py): its own app
name, its own function, its own URL. Redeploying or breaking this file
cannot affect the working fine-tuned deployment in any way.

Usage:
    python -m modal deploy serving/experiments/modal_app_base.py
"""
import os
import pathlib
import subprocess

import modal

HERE = pathlib.Path(__file__).resolve().parent.parent  # serving/ (this file lives in serving/experiments/)
PROXY_AUTH = os.environ.get("CLINIFLOW_PROXY_AUTH") == "1"

image = modal.Image.from_dockerfile(
    str(HERE / "Dockerfile"), context_dir=HERE, add_python="3.12"
).entrypoint([])  # see modal_app_adapter.py for why this is needed

app = modal.App("cliniflow-vllm-base", image=image)


@app.function(
    gpu="T4",
    timeout=60 * 30,
    scaledown_window=60 * 10,
    max_containers=1,
)
# Optional: deploy with CLINIFLOW_PROXY_AUTH=1 to require a Modal proxy-auth token (see serving/README.md).
@modal.web_server(port=8000, startup_timeout=600, requires_proxy_auth=PROXY_AUTH)
def serve_base():
    # Runs vLLM directly, bypassing start_server.sh entirely: no adapter, no
    # --enable-lora, nothing for it to check or attach. --served-model-name
    # gives the "model" field a short name for the eval script to request.
    cmd = (
        "PATH=/usr/bin:/bin:$PATH python3 -m vllm.entrypoints.openai.api_server "
        "--model unsloth/Llama-3.2-3B-Instruct "
        "--served-model-name base-model "
        "--dtype half --max-model-len 4096 "
        "--host 0.0.0.0 --port 8000"
    )
    subprocess.Popen(cmd, shell=True, cwd="/serving")
