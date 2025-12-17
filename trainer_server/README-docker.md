# Silvertongue Trainer Server - Docker Deployment

## Quick Start (with NVIDIA GPU)

1. **Build the image**

```bash
cd trainer_server
sudo docker build -t silvertongue-trainer .
```

2. **Run the container (with GPU and NAS mount)**

```bash
sudo docker run -d \
  --gpus all \
  -p 5050:5050 \
  -v /mnt/nas/silvertongue/array:/mnt/nas/silvertongue/array \
  -v /mnt/nas/silvertongue/models:/mnt/nas/silvertongue/models \
  -e SILVERTONGUE_DATA_DIR=/mnt/nas/silvertongue/array \
  -e SILVERTONGUE_OUTPUT_DIR=/mnt/nas/silvertongue/models \
  -e SILVERTONGUE_BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  --name silvertongue-trainer \
  silvertongue-trainer
```

- `--gpus all` enables NVIDIA GPU access (requires NVIDIA Container Toolkit)
- `-v` mounts NAS data and output directories
- `-e` sets config (can be changed in Portainer UI)

3. **Deploy with Portainer**

- Use the Portainer UI to create a new container/service
- Set the image: `silvertongue-trainer`
- Map port 5050
- Add GPU support (NVIDIA runtime)
- Add environment variables for config
- Mount NAS volumes as above

4. **Check health**

Visit: `http://<dgx-spark-ip>:5050/health`

## Notes
- Requires NVIDIA GPU and drivers, NVIDIA Container Toolkit, and CUDA 12.1+.
- All config can be set via environment variables or Docker/Portainer UI.
- NAS must be mounted inside the container at the same path as the host.
- For updates, just rebuild and redeploy the container.

## Example Portainer Stack YAML

```yaml
version: '3.8'
services:
  silvertongue-trainer:
    image: silvertongue-trainer:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "5050:5050"
    environment:
      SILVERTONGUE_DATA_DIR: /mnt/nas/silvertongue/array
      SILVERTONGUE_OUTPUT_DIR: /mnt/nas/silvertongue/models
      SILVERTONGUE_BASE_MODEL: meta-llama/Llama-3.2-3B-Instruct
    volumes:
      - /mnt/nas/silvertongue/array:/mnt/nas/silvertongue/array
      - /mnt/nas/silvertongue/models:/mnt/nas/silvertongue/models
```

---

**Now you can deploy, update, and manage the trainer server easily with Docker and Portainer!**
