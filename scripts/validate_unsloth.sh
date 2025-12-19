#!/bin/bash
#
# Validate Unsloth Installation on NVIDIA NIM Container
# Reference: https://build.nvidia.com/spark/unsloth/instructions
#
# This runs the official NVIDIA validation test to ensure Unsloth
# is working correctly before starting the training server.
#

export CONTAINER_NAME="unsloth_validation"

# Validate NGC_API_KEY
if [ -z "$NGC_API_KEY" ]; then
    echo "ERROR: NGC_API_KEY is required"
    echo "Get your API key at: https://org.ngc.nvidia.com/setup/api-keys"
    exit 1
fi

echo "=========================================="
echo "Unsloth Validation Test"
echo "=========================================="

# Login to NGC
echo "[INFO] Logging into NGC registry..."
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin

# Stop any existing validation container
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# Run the validation
docker run \
  --name "$CONTAINER_NAME" \
  --rm \
  --gpus all \
  --ipc host \
  -e HF_TOKEN="$HF_TOKEN" \
  nvcr.io/nim/unsloth/unsloth:1.0.0 \
  bash -c '
    echo "=========================================="
    echo "Step 4: Installing dependencies..."
    echo "=========================================="
    pip install transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1"
    pip install --no-deps unsloth unsloth_zoo bitsandbytes

    echo "=========================================="
    echo "Step 5: Downloading validation test..."
    echo "=========================================="
    curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/unsloth/assets/test_unsloth.py

    echo "=========================================="
    echo "Step 6: Running validation test..."
    echo "=========================================="
    python test_unsloth.py

    echo "=========================================="
    echo "Validation Complete!"
    echo "=========================================="
    echo ""
    echo "If you saw:"
    echo "  - Unsloth: Will patch your computer to enable 2x faster free finetuning"
    echo "  - Training progress bars with loss decreasing"
    echo "  - Final training metrics"
    echo ""
    echo "Then Unsloth is working correctly!"
    echo "You can now run: bash scripts/launch_lora_trainer.sh"
  '
