#!/bin/bash

# Use Llama 3.1 8B which HAS LoRA support
export MODEL_HANDLE="meta-llama/Llama-3.1-8B-Instruct"
export LORA_ADAPTERS_PATH="/mnt/silvertongueArray/lora_adapters"
export LOG_DIR="${LOG_BASE:-/var/log/llm_services}/vllm/log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# =============================================================================
# Auto-discover LoRA adapters from versioned directory structure
# Structure: /lora_adapters/actor_name/v1/, /lora_adapters/actor_name/v2/, etc.
# Registers as: actor_name/v1, actor_name/v2, etc.
# =============================================================================
LORA_ARGS=()
if [ -d "$LORA_ADAPTERS_PATH" ]; then
    echo "Scanning for LoRA adapters in $LORA_ADAPTERS_PATH..."
    shopt -s nullglob
    for actor_dir in "$LORA_ADAPTERS_PATH"/*/; do
        if [ -d "$actor_dir" ]; then
            actor_name="$(basename "$actor_dir")"
            
            # Skip non-directory files like adapter_book.json
            [ -f "$LORA_ADAPTERS_PATH/$actor_name" ] && continue
            
            # Check for versioned structure (v1, v2, etc.)
            found_versions=false
            for version_dir in "$actor_dir"v*/; do
                if [ -d "$version_dir" ] && [ -f "$version_dir/adapter_config.json" ]; then
                    version="$(basename "$version_dir")"
                    adapter_id="${actor_name}/${version}"
                    # Path inside Docker container
                    container_path="/lora_adapters/${actor_name}/${version}"
                    LORA_ARGS+=("--lora-modules" "${adapter_id}=${container_path}")
                    echo "  [INFO] Found LoRA adapter: ${adapter_id}"
                    found_versions=true
                fi
            done
            
            # Also check for legacy flat structure (adapter_config.json directly in actor dir)
            if [ "$found_versions" = false ] && [ -f "$actor_dir/adapter_config.json" ]; then
                container_path="/lora_adapters/${actor_name}"
                LORA_ARGS+=("--lora-modules" "${actor_name}=${container_path}")
                echo "  [INFO] Found LoRA adapter (legacy flat): ${actor_name}"
            fi
        fi
    done
    echo "Total LoRA adapters found: $((${#LORA_ARGS[@]} / 2))"
else
    echo "Warning: LoRA adapters directory not found: $LORA_ADAPTERS_PATH"
fi

nohup docker run \
    --name vllm_llm_server \
    --rm \
    --gpus all \
    --ipc host \
    --network host \
    -e HF_TOKEN=$HF_TOKEN \
    -e MODEL_HANDLE="$MODEL_HANDLE" \
    -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
    -v $LORA_ADAPTERS_PATH:/lora_adapters \
    nvcr.io/nvidia/vllm:25.11-py3 \
    vllm serve "$MODEL_HANDLE" \
    --enable-lora \
    --max-lora-rank 64 \
    "${LORA_ARGS[@]}" \
    --max-model-len 32768 \
    --max-num-seqs 3 \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8355 \
    --trust-remote-code \
  > "$LOG_DIR/vllm_server_llama31_8b.log" 2>&1 &

echo "vLLM server starting with LoRA support..."
echo "Model: $MODEL_HANDLE"
echo "Logs: $LOG_DIR/vllm_server_llama31_8b.log"
echo ""
echo "LoRA Adapters loaded: $((${#LORA_ARGS[@]} / 2))"
echo "To use a LoRA adapter in API calls:"
echo '  model="steve/v1"  # Use versioned adapter name'
echo ""
echo "Following logs..."
sleep 2
tail -F "$LOG_DIR/vllm_server_llama31_8b.log"
