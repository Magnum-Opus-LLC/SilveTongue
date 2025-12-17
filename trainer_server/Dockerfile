# Silvertongue LoRA Trainer Server Dockerfile
# For NVIDIA GPU (CUDA 12.1+), Hugging Face, Flask, LoRA fine-tuning

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and server code
COPY requirements.txt ./
COPY server.py ./

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5050

# Environment variables for config (can be set in Portainer)
ENV SILVERTONGUE_DATA_DIR=/mnt/nas/silvertongue/array \
    SILVERTONGUE_OUTPUT_DIR=/mnt/nas/silvertongue/models \
    SILVERTONGUE_BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct

# Entrypoint
CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "5050"]
