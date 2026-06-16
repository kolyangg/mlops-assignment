#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

if [[ -f .env ]]; then
    set -a
    source .env
    set +a
fi

MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"

exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --enable-prefix-caching
