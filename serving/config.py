"""Serving settings, read from environment variables.

Nothing machine-specific is hardcoded. Defaults suit a local or Colab test;
override them on the AWS instance or in Docker with -e flags.
"""
import os

# Full-precision base model served once. The adapter was trained on the 4-bit
# Unsloth build; confirm this repo is accessible before the first real run.
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Llama-3.2-3B-Instruct")

# Folder holding adapter_model.safetensors, adapter_config.json and system_prompt.txt.
ADAPTER_DIR = os.environ.get("ADAPTER_DIR", "./adapter")

# The name clients send in the "model" field. The CliniFlow deployment (modal_app.py) serves "cliniflow-soap";
# for the unmerged adapter server (start_server.sh, experiments/) set MODEL_NAME to the adapter name below.
MODEL_NAME = os.environ.get("MODEL_NAME", "cliniflow-soap")

# The name the unmerged adapter server (start_server.sh) registers its LoRA adapter under.
ADAPTER_NAME = os.environ.get("ADAPTER_NAME", "clinical-json-extractor")

# Must be >= the adapter rank (16 for config A).
MAX_LORA_RANK = int(os.environ.get("MAX_LORA_RANK", "16"))

# Longest input + output in tokens. Training used max_seq 4096.
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))

# Fraction of GPU memory vLLM may claim.
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"))

# Where the OpenAI-compatible server listens.
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# Where clients send requests.
SERVER_URL = os.environ.get("SERVER_URL", f"http://localhost:{PORT}")


def auth_headers():
    """Modal proxy-auth token headers, if the endpoint was deployed with requires_proxy_auth=True."""
    key, secret = os.environ.get("MODAL_PROXY_TOKEN_ID"), os.environ.get("MODAL_PROXY_TOKEN_SECRET")
    return {"Modal-Key": key, "Modal-Secret": secret} if key and secret else {}
