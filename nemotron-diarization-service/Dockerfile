# ===============================================================================
# NEMOTRON-3 DIARIZATION SERVICE - PRODUCTION DOCKERFILE
# ===============================================================================
# Hỗ trợ: NVIDIA CUDA GPU runtime
# Khởi tạo: docker build -t nemotron-diarization-service .
# Chạy GPU: docker run --gpus all -p 8000:8000 nemotron-diarization-service
# ===============================================================================

FROM python:3.10-slim

# Cài đặt thư viện hệ thống cần thiết cho âm thanh và compile
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

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
