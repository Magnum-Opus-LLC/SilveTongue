#!/bin/bash
#
# Launch LoRA Training Server for Silvertongue
# Uses NVIDIA PyTorch container with Unsloth for LoRA training
#
# Reference: https://build.nvidia.com/spark/unsloth/instructions
#

export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export LORA_OUTPUT_DIR="${LORA_OUTPUT_DIR:-/mnt/silvertongueArray/lora_adapters}"
export DATASET_DIR="${DATASET_DIR:-/mnt/silvertongueArray/array/data}"
export PORT="${PORT:-5050}"
export CONTAINER_NAME="silvertongue_trainer"

# Use PyTorch 25.11 for Blackwell (B200/GB10) support
# Check: https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html
PYTORCH_IMAGE="${PYTORCH_IMAGE:-nvcr.io/nvidia/pytorch:25.11-py3}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$HF_TOKEN" ]; then
    echo "[WARN] HF_TOKEN not set - may fail to download gated models"
fi

echo "=========================================="
echo "Silvertongue LoRA Training Server"
echo "=========================================="
echo "Base Model:  $BASE_MODEL"
echo "LoRA Output: $LORA_OUTPUT_DIR"
echo "Dataset Dir: $DATASET_DIR"
echo "Port:        $PORT"
echo "Container:   $PYTORCH_IMAGE"
echo "=========================================="

# Create output directories
mkdir -p "$LORA_OUTPUT_DIR"
mkdir -p "$DATASET_DIR"

# Stop existing container if running
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "[INFO] Starting PyTorch container with Unsloth..."

docker run \
  --name "$CONTAINER_NAME" \
  --rm \
  --gpus all \
  --ipc host \
  --network host \
  -e HF_TOKEN="$HF_TOKEN" \
  -e BASE_MODEL="$BASE_MODEL" \
  -e PORT="$PORT" \
  -v "$LORA_OUTPUT_DIR":/lora_adapters \
  -v "$DATASET_DIR":/datasets \
  -v "$SCRIPT_DIR":/app \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -w /app \
  "$PYTORCH_IMAGE" \
  bash -c '
    echo "=========================================="
    echo "Installing Unsloth and dependencies..."
    echo "=========================================="

    # Install dependencies (per NVIDIA Unsloth instructions)
    pip install -q transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1"
    pip install -q --no-deps unsloth==2025.12.8 transformers==4.57.3 unsloth_zoo bitsandbytes psutil

    # Install Flask for the training server
    pip install -q flask flask-cors

    FILE="/app/unsloth_compiled_cache/UnslothSFTTrainer.py"

    if [ -f "$FILE" ]; then
      if ! grep -qE "^[[:space:]]*import[[:space:]]+psutil[[:space:]]*$" "$FILE"; then
        sed -i "2i import psutil" "$FILE"
        echo "1"
      else
        echo "2"
      fi
    else
      echo "3"
    fi

    echo "=========================================="
    echo "Starting LoRA Training Server on port $PORT..."
    echo "=========================================="

    python lora_trainer_server.py \
      --port $PORT \
      --base-model "$BASE_MODEL" \
      --lora-output /lora_adapters \
      --dataset-dir /datasets
  '
