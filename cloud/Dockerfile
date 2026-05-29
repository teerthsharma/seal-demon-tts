# DemonTTS Cloud GPU Container
# Supports NVIDIA GPUs via CUDA 12.1
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev \
    git wget curl ffmpeg libsndfile1 \
    build-essential cmake pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements-cloud.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements-cloud.txt

# Install huggingface-cli for model caching
RUN pip3 install huggingface-hub[cli]

# Pre-download SpeechT5 to avoid runtime download
RUN python3 -c "from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan; \
    SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts'); \
    SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')"

# Working dir
WORKDIR /workspace/demon-tts
COPY . /workspace/demon-tts/

# Cloud entrypoints
RUN chmod +x /workspace/demon-tts/cloud/entrypoint.sh

EXPOSE 8000
CMD ["/workspace/demon-tts/cloud/entrypoint.sh"]
