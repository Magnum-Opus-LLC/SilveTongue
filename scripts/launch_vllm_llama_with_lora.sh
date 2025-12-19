#!/bin/bash

# Use Llama 3.1 8B which HAS LoRA support
export MODEL_HANDLE="meta-llama/Llama-3.1-8B-Instruct"
export LORA_ADAPTERS_PATH="/mnt/nas/silvertongue/lora_adapters"
export LOG_DIR="/home/rhmaomao/myServer/llm_services/vllm/log"

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

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
echo "To add LoRA adapters dynamically:"
echo "  cp -r /path/to/actor_lora_adapter $LORA_ADAPTERS_PATH/actor_name"
echo ""
echo "Then in your code, use:"
echo '  extra_body={"lora_name": "actor_name"}'
echo ""
echo "Following logs..."
sleep 2
tail -F "$LOG_DIR/vllm_server_llama31_8b.log"
