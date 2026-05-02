#!/bin/bash

export BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct-AWQ"   # quantized Qwen VL
export HOST_MODEL_ROOT="${HOST_MODEL_ROOT:-/data/llm_models}"
export HOST_LORA_ROOT="/mnt/silvertongueArray/lora_adapters"
export LOG_DIR="${LOG_BASE:-/var/log/llm_services}/nvcr_hosting/qwen2.5vl-vllm/log"
export PORT=8355

# Ensure directories exist
mkdir -p "$HOST_MODEL_ROOT"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/vllm_qwen2.5vl_awq.log"

if [ -z "$HF_TOKEN" ]; then
    echo "[ERROR] HF_TOKEN environment variable is not set."
    exit 1
fi

echo "Starting vLLM with $BASE_MODEL..."

nohup docker run \
  --name vllm_llm_server \
  --rm \
  --gpus all \
  --ipc host \
  --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e BASE_MODEL="$BASE_MODEL" \
  -e PORT="$PORT" \
  -e PYTORCH_ALLOC_CONF="expandable_segments:True" \
  -v "$HOST_MODEL_ROOT":/models \
  -v "$HOST_LORA_ROOT":/lora_adapters:ro \
  nvcr.io/nvidia/vllm:25.11-py3 \
  bash -lc '
    set -euo pipefail

    # Use a clean model directory name
    SAFE_NAME="$(echo "$BASE_MODEL" | tr "/" "_" | tr ":" "_")"
    MODEL_DIR="/models/${SAFE_NAME}"

    # Check if model already exists (look for model files, not just config)
    if [ -d "$MODEL_DIR" ] && [ "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
      echo "[INFO] Model already present at $MODEL_DIR"
    else
      echo "[INFO] Downloading $BASE_MODEL -> $MODEL_DIR"
      mkdir -p "$MODEL_DIR"
      huggingface-cli download "$BASE_MODEL" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
    fi

    # Auto-discovery for LoRA adapters (versioned structure)
    # Structure: /lora_adapters/actor_name/v1/, /lora_adapters/actor_name/v2/, etc.
    # Registers as: actor_name/v1, actor_name/v2, etc.
    LORA_ARGS=()
    if [ -d "/lora_adapters" ]; then
      shopt -s nullglob
      for d in /lora_adapters/*; do
        if [ -d "$d" ]; then
          name="$(basename "$d")"
          # Skip non-adapter files like adapter_book.json
          [ "$name" = "adapter_book.json" ] && continue
          
          # Check for versioned structure (v1, v2, etc.)
          found_versions=false
          for v in "$d"/v*; do
            if [ -d "$v" ] && [ -f "$v/adapter_config.json" ]; then
              version="$(basename "$v")"
              adapter_id="${name}/${version}"
              LORA_ARGS+=("--lora-modules" "${adapter_id}=${v}")
              echo "[INFO] Found LoRA adapter: ${adapter_id}"
              found_versions=true
            fi
          done
          
          # Also check for legacy flat structure (adapter_config.json directly in actor dir)
          if [ "$found_versions" = false ] && [ -f "$d/adapter_config.json" ]; then
            LORA_ARGS+=("--lora-modules" "${name}=${d}")
            echo "[INFO] Found LoRA adapter (legacy flat): ${name}"
          fi
        fi
      done
      echo "[INFO] Total LoRA adapters found: $((${#LORA_ARGS[@]} / 2))"
    fi

    # Serve OpenAI-compatible API (optimized for Blackwell B200)
    vllm serve "$MODEL_DIR" \
      --host 0.0.0.0 --port "$PORT" \
      --served-model-name qwen2_5vl \
      --trust-remote-code \
      --dtype bfloat16 \
      --quantization awq_marlin \
      --kv-cache-dtype auto \
      --gpu-memory-utilization 0.95 \
      --max-model-len 32768 \
      --max-num-seqs 8 \
      --enable-chunked-prefill \
      --limit-mm-per-prompt "{\"image\":2,\"video\":0}" \
      --enable-lora \
      --max-lora-rank 64 \
      ${LORA_ARGS[@]:+"${LORA_ARGS[@]}"} \
      --disable-log-requests
  ' \
  > "$LOG_FILE" 2>&1 &

echo "vLLM server starting on port $PORT..."
echo "Logs: tail -F $LOG_FILE"
sleep 2
tail -F "$LOG_FILE"
