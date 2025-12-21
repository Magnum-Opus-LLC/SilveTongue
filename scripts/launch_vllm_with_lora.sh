#!/bin/bash

# Llama 4 Maverick (17B-128E MoE) - Native multimodal with LoRA support
# 17B activated params, 400B total | 1M context | FP8 quantized
# Best-in-class for scholarly research: diagrams, charts, tables, equations
# See: https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
export MODEL_HANDLE="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
export LORA_ADAPTERS_PATH="/mnt/silvertongueArray/lora_adapters"
export LOG_DIR="/home/rhmaomao/myServer/llm_services/vllm/log"

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
    --max-model-len 80000 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.8 \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port 8355 \
    --trust-remote-code \
    --limit-mm-per-prompt '{"image": 5}' \
  > "$LOG_DIR/vllm_server_llama4_maverick.log" 2>&1 &

echo "vLLM server starting with Llama 4 Maverick..."
echo "Model: $MODEL_HANDLE"
echo "Architecture: 17B activated (400B total MoE) | FP8 | Native Multimodal"
echo "Logs: $LOG_DIR/vllm_server_llama4_maverick.log"
echo ""
echo "Capabilities:"
echo "  ✓ Native multimodal (text + images with early fusion)"
echo "  ✓ Scientific papers (DocVQA: 94.4, ChartQA: 90.0)"
echo "  ✓ Diagram reasoning (MMMU Pro: 59.6)"
echo "  ✓ 1M context window (80K configured)"
echo "  ✓ LoRA fine-tuning for personalized learning"
echo "  ✓ 12 languages + multilingual support"
echo ""
echo "LoRA Adapters loaded: $((${#LORA_ARGS[@]} / 2))"
echo "To use a LoRA adapter in API calls:"
echo '  model="steve/v1"  # Use versioned adapter name'
echo ""
echo "API usage (text + up to 5 images):"
echo '  content=[{"type":"text","text":"..."},{"type":"image_url","image_url":{"url":"..."}}]'
echo ""
echo "Following logs..."
sleep 2
tail -F "$LOG_DIR/vllm_server_llama4_maverick.log"
