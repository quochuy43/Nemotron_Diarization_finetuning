# ===============================================================================
# NEMOTRON-3 DIARIZATION SERVICE - PRODUCTION DOCKERFILE (CUDA GPU)
# ===============================================================================
# Base image: NVIDIA CUDA 12.1 runtime (cần thiết để torch.cuda hoạt động)
# Build:  docker build -t nemotron-diarization-service .
# Run:    docker run --gpus all -p 8000:8000 nemotron-diarization-service
# ===============================================================================

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Cài đặt Python 3.10 và thư viện hệ thống cho âm thanh
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Cài đặt PyTorch với CUDA 12.1
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy và cài đặt requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn và checkpoint xuất xưởng
COPY checkpoints/ ./checkpoints/
COPY server/ ./server/
COPY client/ ./client/
COPY docs/ ./docs/

# Cấu hình biến môi trường
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Khởi chạy FastAPI WebSocket Server
CMD ["python", "server/websocket_server.py", "--host", "0.0.0.0", "--port", "8000"]
