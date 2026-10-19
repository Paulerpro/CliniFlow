#!/usr/bin/env bash
# Starts the vLLM OpenAI-compatible server with the CliniFlow adapter enabled.
# Used unchanged by Docker (local or AWS) and by the Colab notebook.
set -euo pipefail

# Settings come from config.py's environment variables; these are the defaults.
: "${BASE_MODEL:=unsloth/Llama-3.2-3B-Instruct}"
: "${ADAPTER_DIR:=./adapter}"
: "${ADAPTER_NAME:=clinical-json-extractor}"
: "${MAX_LORA_RANK:=16}"
: "${MAX_MODEL_LEN:=4096}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"

if [ ! -f "${ADAPTER_DIR}/adapter_config.json" ]; then
  echo "ERROR: no adapter found in ${ADAPTER_DIR} (expected adapter_config.json)" >&2
  exit 1
fi

exec python3 -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" \
  --dtype half \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --lora-modules "${ADAPTER_NAME}=${ADAPTER_DIR}" \
  --host "${HOST}" \
  --port "${PORT}"
