# Deployment with Docker (CPU and GPU)

This guide shows how to run the Music-Therapy app with TensorFlow and DeepFace in production using Docker.

## Overview
- Use `Dockerfile.cpu` for a CPU-only deployment.
- Use `Dockerfile.gpu` for GPU deployments (host must have NVIDIA drivers and `nvidia-docker2` / `--gpus` support).
- `requirements-optional.txt` contains heavy TF-backed deps (`tensorflow`, `tf-keras`, `deepface`).

## Build & Run (CPU)

Build the image:
```bash
docker build -t music-therapy:cpu -f Dockerfile.cpu .
```

Run the container (expose Streamlit port 8501):
```bash
docker run --rm -p 8501:8501 --name music-therapy-cpu music-therapy:cpu
```

## Build & Run (GPU)

Ensure the host has NVIDIA drivers and the `nvidia` runtime enabled.

Build the GPU image:
```bash
docker build -t music-therapy:gpu -f Dockerfile.gpu .
```

Run with GPU access:
```bash
docker run --gpus all -p 8501:8501 --name music-therapy-gpu music-therapy:gpu
```

## Verify TensorFlow & DeepFace inside the container

Start an interactive shell in the running container:
```bash
docker exec -it music-therapy-cpu /bin/bash
# or for GPU
docker exec -it music-therapy-gpu /bin/bash
```

Then run these checks:
```bash
python -c "import tensorflow as tf; print('TF', tf.__version__)"
python -c "from deepface import DeepFace; print('DeepFace OK')"
```

## Notes & Compatibility
- Match the `tensorflow` version in `requirements-optional.txt` to the CUDA/cuDNN versions of your GPU image (see TensorFlow release notes).
- If using the GPU image, pick a base `nvidia/cuda:<version>-cudnn<ver>-runtime-ubuntu<ver>` that matches TF compatibility.
- First-time image builds may take several minutes as models and packages download.
- If you prefer lighter deployment, host TensorFlow/DeepFace as a separate microservice (FastAPI) and call it from the Streamlit app.

## Troubleshooting
- Out of memory: increase host memory or use swap; try smaller batch sizes.
- Slow startup: pre-warm DeepFace by calling `emotion_detector.setup_deepface()` once (the repo contains this helper).
- Streamlit Cloud: this approach requires a host with custom Docker support (not available on Streamlit Community Cloud free tier).

