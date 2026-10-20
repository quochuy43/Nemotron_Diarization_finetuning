#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
PRODUCTION ARCHITECTURE 1: WEBSOCKET STREAMING DIARIZATION SERVER (FASTAPI)
===============================================================================
Mục đích:
  - Cung cấp dịch vụ Streaming Speaker Diarization thời gian thực chuẩn SLA 0.32s
    thông qua giao thức WebSocket và cơ chế Ring Buffer.
  - Hỗ trợ đa phiên đồng thời (Multi-session concurrency) với cấu trúc bộ nhớ
    cô lập cho từng cuộc gọi / microphone.
  - Nhận luồng âm thanh PCM 16kHz mono (từ Browser Web Audio API, Twilio, FreeSWITCH,
    hoặc Python client).
  - Tích hợp bộ đệm xoay vòng (Ring Buffer) 15–30s và thuật toán Hungarian Matching
    để khóa kênh người nói liên tục, chống đảo kênh (Zero Permutation Swapping).
  - Phản hồi sự kiện theo thời gian thực (real-time timestamps, active speakers, confidence).
===============================================================================
"""

import os
import sys
import time
import json
import uuid
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

print("[*] Đang khởi động WebSocket Streaming Server (vui lòng chờ trong giây lát)...", flush=True)

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForAudioFrameClassification

# Cấu hình đường dẫn nội bộ
current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir.parent
for p in [str(scripts_dir), str(scripts_dir / "02_baseline_eval"), str(scripts_dir / "03_training")]:
    if p not in sys.path:
        sys.path.insert(0, p)


def resolve_file_path(path_str: str) -> str:
    """Tự động chuẩn hóa và tìm kiếm file trên Windows / Colab / Drive."""
    if not path_str:
        return path_str
    if Path(path_str).exists():
        return path_str
    norm_path = path_str.replace("\\", "/")
    if "data/" in norm_path:
        sub_rel = norm_path[norm_path.find("data/"):]
        candidates = [
            Path.cwd() / sub_rel,
            Path(__file__).resolve().parent.parent.parent / sub_rel,
            Path("/content/diarization") / sub_rel,
            Path("/content/drive/MyDrive/VJ_Technologies/Nemotron_Diarization") / sub_rel,
            Path("/content/drive/MyDrive") / sub_rel,
        ]
        for cand in candidates:
            if cand.exists():
                return str(cand)
    return path_str


# ===============================================================================
# 1. CẤU HÌNH HỆ THỐNG MẶC ĐỊNH (PRODUCTION DEFAULTS)
# ===============================================================================

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHUNK_DURATION = 0.24       # 240ms bước nhảy
DEFAULT_LOOKAHEAD_DURATION = 0.08   # 80ms nhìn trước (Tổng SLA = 320ms)
DEFAULT_HISTORY_DURATION = 2.64     # 2.64s FIFO context buffer (264 frames)
DEFAULT_RING_BUFFER_SEC = 20.0      # Giữ 20s âm thanh gần nhất trong RAM
DEFAULT_THRESHOLD = 0.45            # Ngưỡng kích hoạt tiếng nói tối ưu
DEFAULT_MIN_SPEECH_ON = 0.05        # Lọc nhiễu < 50ms
DEFAULT_MIN_GAP_OFF = 0.15          # Nối khoảng ngắt < 150ms


# ===============================================================================
# 2. AUDIO RING BUFFER: BỘ ĐỆM XOAY VÒNG CHO TỪNG PHIÊN (PER-SESSION BUFFER)
# ===============================================================================

class AudioRingBuffer:
    """
    Bộ đệm âm thanh xoay vòng (Circular Ring Buffer) an toàn cho từng phiên kết nối:
    - Nhận dữ liệu âm thanh dạng bytes (PCM 16-bit mono 16kHz) hoặc float32.
    - Lưu trữ lịch sử tối đa max_seconds (mặc định 20 giây).
    - Cung cấp cửa sổ trượt: [History (tối đa 2.64s) | Chunk hiện tại (0.24s) | Lookahead (0.08s)].
    """
    def __init__(self, sample_rate: int = DEFAULT_SAMPLE_RATE, max_seconds: float = DEFAULT_RING_BUFFER_SEC):
        self.sample_rate = sample_rate
        self.capacity = int(round(sample_rate * max_seconds))
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_idx = 0
        self.total_samples_received = 0
        self.last_emitted_sample_idx = 0

    def append_pcm16_bytes(self, pcm_bytes: bytes) -> int:
        """Nhận mảng byte PCM 16-bit little endian và chuyển đổi sang float32 [-1.0, 1.0]."""
        if not pcm_bytes:
            return 0
        # Chuyển bytes thành int16
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = (audio_int16.astype(np.float32) / 32768.0).copy()
        return self.append_float32(audio_float)

    def append_float32(self, samples: np.ndarray) -> int:
        """Nạp các mẫu float32 vào ring buffer."""
        n = len(samples)
        if n == 0:
            return 0

        # Nếu lượng samples mới vượt quá capacity, chỉ lấy phần đuôi
        if n >= self.capacity:
            self.buffer[:] = samples[-self.capacity:]
            self.write_idx = 0
        else:
            end_idx = (self.write_idx + n) % self.capacity
            if self.write_idx + n <= self.capacity:
                self.buffer[self.write_idx : self.write_idx + n] = samples
            else:
                part1 = self.capacity - self.write_idx
                self.buffer[self.write_idx :] = samples[:part1]
                self.buffer[: end_idx] = samples[part1:]
            self.write_idx = end_idx

        self.total_samples_received += n
        return n

    def get_linear_history(self, num_samples: int) -> np.ndarray:
        """Lấy tuyến tính N samples âm thanh gần nhất từ ring buffer."""
        available = min(self.total_samples_received, self.capacity)
        num_samples = min(num_samples, available)
        if num_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        start_pos = (self.write_idx - num_samples) % self.capacity
        if start_pos + num_samples <= self.capacity:
            return self.buffer[start_pos : start_pos + num_samples].copy()
        else:
            part1 = self.capacity - start_pos
            part2 = num_samples - part1
            return np.concatenate([self.buffer[start_pos:], self.buffer[:part2]]).copy()

    def has_enough_for_chunk(self, chunk_samples: int, lookahead_samples: int) -> bool:
        """Kiểm tra xem đã có đủ mẫu để suy luận chunk tiếp theo chưa."""
        samples_since_last = self.total_samples_received - self.last_emitted_sample_idx
        # Cần ít nhất 1 chunk mới + phần lookahead
        return samples_since_last >= (chunk_samples + lookahead_samples)


# ===============================================================================
# 3. STREAMING DIARIZATION WORKER (QUẢN LÝ SUY LUẬN & KHÓA KÊNH HUNGARIAN)
# ===============================================================================

class DiarizationSession:
    """
    Quản lý trạng thái phân đoạn người nói của 1 phiên làm việc:
    - Giữ Ring Buffer âm thanh.
    - Duy trì ma trận xác suất lịch sử (full_probs_so_far) để thực hiện Hungarian Matching.
    - Đếm số lượng chunks đã emit và tính thời gian biểu (timestamps).
    """
    def __init__(
        self,
        session_id: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_duration: float = DEFAULT_CHUNK_DURATION,
        lookahead_duration: float = DEFAULT_LOOKAHEAD_DURATION,
        history_duration: float = DEFAULT_HISTORY_DURATION,
        threshold: float = DEFAULT_THRESHOLD
    ):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.lookahead_duration = lookahead_duration
        self.history_duration = history_duration
        self.threshold = threshold

        self.chunk_samples = int(round(chunk_duration * sample_rate))         # 3,840 samples (240ms)
        self.lookahead_samples = int(round(lookahead_duration * sample_rate)) # 1,280 samples (80ms)
        self.history_samples = int(round(history_duration * sample_rate))     # 42,240 samples (2.64s)
        self.total_buffer_samples = self.chunk_samples + self.lookahead_samples # 5,120 samples (320ms)

        self.ring_buffer = AudioRingBuffer(sample_rate=sample_rate)
        self.full_probs_so_far: Optional[np.ndarray] = None
        self.chunk_count = 0
        self.created_at = time.time()
        self.active_speaker_history: List[str] = []

    def get_inference_window(self) -> Tuple[np.ndarray, int, int]:
        """
        Trích xuất cửa sổ trượt: [History | Current Chunk | Lookahead]
        Trả về: (audio_slice, chunk_offset_samples, actual_chunk_len)
        """
        # Điểm bắt đầu của chunk cần emit
        cur_start = self.ring_buffer.last_emitted_sample_idx
        history_start = max(0, cur_start - self.history_samples)
        cur_end = cur_start + self.chunk_samples
        buffer_end = cur_end + self.lookahead_samples

        # Lấy âm thanh từ ring buffer tương ứng khoảng [history_start : buffer_end]
        samples_needed = buffer_end - history_start
        audio_slice = self.ring_buffer.get_linear_history(
            num_samples=(self.ring_buffer.total_samples_received - history_start)
        )
        if len(audio_slice) > samples_needed:
            audio_slice = audio_slice[:samples_needed]

        # Đệm zero nếu không đủ 320ms
        if len(audio_slice) < self.total_buffer_samples:
            pad = self.total_buffer_samples - len(audio_slice)
            audio_slice = np.pad(audio_slice, (0, pad), mode="constant")

        chunk_offset_samples = cur_start - history_start
        return audio_slice, chunk_offset_samples, self.chunk_samples

    def align_and_update_probs(self, raw_probs: np.ndarray, chunk_offset_samples: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Thực hiện Hungarian Matching để khóa kênh cố định và trích xuất phân đoạn hoạt động.
        """
        chunk_offset_frames = int(round((chunk_offset_samples / self.sample_rate) * 100))
        emitted_frames = int(round((self.chunk_samples / self.sample_rate) * 100))

        aligned_probs = raw_probs.copy()

        # 1. Khóa kênh với lịch sử
        if self.full_probs_so_far is not None and chunk_offset_frames > 0:
            overlap_len = min(chunk_offset_frames, len(self.full_probs_so_far), len(aligned_probs))
            if overlap_len > 0:
                prev_overlap = self.full_probs_so_far[-overlap_len:]
                curr_overlap = aligned_probs[chunk_offset_frames - overlap_len : chunk_offset_frames]

                cost_matrix = np.zeros((8, 8), dtype=np.float32)
                for i in range(8):
                    for j in range(8):
                        diff = curr_overlap[:, i] - prev_overlap[:, j]
                        cost_matrix[i, j] = np.mean(diff ** 2) + (0.0 if i == j else 1e-4)

                curr_idx, tracked_idx = linear_sum_assignment(cost_matrix)
                perm = np.zeros(8, dtype=int)
                for c, t in zip(curr_idx, tracked_idx):
                    perm[t] = c
                aligned_probs = aligned_probs[:, perm]

        # 2. Cắt phần 240ms của chunk hiện tại
        valid_chunk = aligned_probs[chunk_offset_frames : chunk_offset_frames + emitted_frames]
        if len(valid_chunk) == 0:
            valid_chunk = np.zeros((emitted_frames, 8), dtype=np.float32)

        # 3. Cập nhật lịch sử xác suất (giữ tối đa 500 frames ~ 5 giây để tối ưu bộ nhớ)
        if self.full_probs_so_far is None:
            self.full_probs_so_far = valid_chunk.copy()
        else:
            self.full_probs_so_far = np.concatenate([self.full_probs_so_far, valid_chunk], axis=0)
            if len(self.full_probs_so_far) > 500:
                self.full_probs_so_far = self.full_probs_so_far[-500:]

        # 4. Đánh giá người nói tích cực trong 240ms này
        mean_probs = np.mean(valid_chunk, axis=0)
        active_speakers = []
        for spk_idx in range(8):
            prob = float(mean_probs[spk_idx])
            if prob >= self.threshold:
                spk_name = f"SPEAKER_{spk_idx:02d}"
                active_speakers.append({
                    "speaker_id": spk_name,
                    "channel_index": spk_idx,
                    "confidence": round(prob, 4)
                })
                if spk_name not in self.active_speaker_history:
                    self.active_speaker_history.append(spk_name)

        # Cập nhật con trỏ đã xử lý
        self.ring_buffer.last_emitted_sample_idx += self.chunk_samples
        self.chunk_count += 1

        return valid_chunk, active_speakers


# ===============================================================================
# 4. GLOBAL MODEL MANAGER (QUẢN LÝ TẬP TRUNG MÔ HÌNH TRÊN GPU)
# ===============================================================================

class ModelManager:
    """Singleton quản lý model Nemotron-3 Diarization dùng chung cho mọi WebSocket."""
    _instance = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.model_path = None
        self.lock = asyncio.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def load_model(self, custom_model_path: Optional[str] = None):
        if self.model_loaded:
            return

        print("\n" + "=" * 80)
        print("🚀 ĐANG KHỞI TẠO NEMOTRON-3 STREAMING SERVER (WEBSOCKET + RING BUFFER)")
        print("=" * 80)

        # Tìm kiếm checkpoint
        candidates = [
            custom_model_path or "",
            "/content/drive/MyDrive/VJ_Technologies/Nemotron_Diarization/experiments/checkpoints/train_v1_3_ratio_1_8_swa_model/model_swa_v1_3_SWA.pt",
            "/content/drive/MyDrive/VJ_Technologies/Nemotron_Diarization/experiments/checkpoints/train_v1_3_ratio_1_8_swa_model/model_swa.pt",
            "experiments/checkpoints/train_v1_3_ratio_1_8_swa_model/model_swa_v1_3_SWA.pt",
            "experiments/checkpoints/train_v1_3_ratio_1_8_swa_model/model_swa.pt",
            "D:/VJ_Technologies/diarization/experiments/checkpoints/train_v1_3_ratio_1_8_swa_model/model_swa_v1_3_SWA.pt"
        ]
        resolved = None
        for cand in candidates:
            if cand:
                rp = resolve_file_path(cand)
                if Path(rp).exists():
                    resolved = rp
                    break

        pretrained_id = "nvidia/Nemotron-3-Diarization"
        print(f"[*] Nạp Pretrained Processor & Architecture: {pretrained_id}...")
        self.processor = AutoProcessor.from_pretrained(pretrained_id)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(pretrained_id)

        if resolved and Path(resolved).exists():
            print(f"[*] Nạp trọng số xuất xưởng SWA: {resolved}")
            payload = torch.load(resolved, map_location="cpu")
            state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
            self.model.load_state_dict(state_dict, strict=False)
            self.model_path = resolved
        else:
            print("[!] Cảnh báo: Sử dụng Pretrained base weights (không tìm thấy checkpoint v1.3 local).")

        self.model.to(self.device)
        self.model.eval()

        # Warm-up GPU 3 chunks để ổn định CUDA kernel
        print(f"[*] Thiết bị suy luận: {self.device.upper()}")
        dummy_audio = np.zeros(5120, dtype=np.float32)
        with torch.inference_mode():
            for _ in range(3):
                inp = self.processor(dummy_audio, sampling_rate=DEFAULT_SAMPLE_RATE, return_tensors="pt").to(self.device)
                _ = self.model(**inp).logits
            if self.device == "cuda":
                torch.cuda.synchronize()

        self.model_loaded = True
        print("[+] MODEL STREAMING ĐÃ SẴN SÀNG PHỤC VỤ (WARM-UP HOÀN TẤT)!\n")

    def forward_audio_slice(self, audio_slice: np.ndarray) -> Tuple[np.ndarray, float]:
        """Thực hiện 1 bước suy luận trên chunk âm thanh."""
        if not self.model_loaded:
            raise RuntimeError("Model chưa được nạp!")

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            inputs = self.processor(audio_slice, sampling_rate=DEFAULT_SAMPLE_RATE, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits # (1, num_frames, 8)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

        if self.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        return probs, latency_ms


# ===============================================================================
# 5. KHỞI TẠO ỨNG DỤNG FASTAPI & SESSION MANAGER
# ===============================================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = ModelManager.get_instance()
    manager.load_model()
    yield

app = FastAPI(
    title="Nemotron-3 Diarization Streaming Server",
    description="Kiến trúc 1: WebSocket + Ring Buffer cho Realtime Diarization (SLA 0.32s)",
    version="1.3.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_sessions: Dict[str, DiarizationSession] = {}


@app.get("/health")
def health_check():
    manager = ModelManager.get_instance()
    return {
        "status": "UP",
        "model_loaded": manager.model_loaded,
        "device": manager.device,
        "active_sessions": len(active_sessions),
        "sla_target_ms": 320.0
    }


# ===============================================================================
# 6. WEBSOCKET ENDPOINT: /ws/diarize (XỬ LÝ DỮ LIỆU STREAMING THỰC TẾ)
# ===============================================================================

@app.websocket("/ws/diarize")
async def websocket_diarize(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None),
    threshold: float = Query(DEFAULT_THRESHOLD)
):
    """
    Kênh WebSocket truyền nhận âm thanh 2 chiều:
    - Client gửi:
        + Binary frame: Mảng bytes PCM 16-bit 16kHz Mono (mỗi gói khoảng 10ms - 240ms).
        + Text frame JSON: {"action": "reset"} hoặc {"action": "ping"}.
    - Server trả về JSON Event ngay khi có chunk 240ms hoàn thành.
    """
    await websocket.accept()
    session_id = client_id or f"sess_{uuid.uuid4().hex[:8]}"
    
    session = DiarizationSession(session_id=session_id, threshold=threshold)
    active_sessions[session_id] = session
    manager = ModelManager.get_instance()

    print(f"[+] Kết nối mới WebSocket: {session_id} (Đang phục vụ {len(active_sessions)} luồng)")

    # Gửi bản tin chào mừng và cấu hình
    await websocket.send_json({
        "type": "connection_established",
        "session_id": session_id,
        "sample_rate": DEFAULT_SAMPLE_RATE,
        "chunk_duration_sec": DEFAULT_CHUNK_DURATION,
        "sla_latency_sec": 0.32,
        "threshold": threshold,
        "message": "Sẵn sàng nhận dữ liệu âm thanh PCM 16-bit 16kHz mono qua WebSocket."
    })

    try:
        while True:
            message = await websocket.receive()
            
            # 1. Nhận gói âm thanh nhị phân (Binary PCM16)
            if "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                session.ring_buffer.append_pcm16_bytes(raw_bytes)

                # 2. Kiểm tra xem đã đủ mẫu để chạy 1 chunk 240ms chưa
                while session.ring_buffer.has_enough_for_chunk(session.chunk_samples, session.lookahead_samples):
                    audio_slice, offset_samples, emitted_len = session.get_inference_window()

                    # Chạy suy luận qua model lock (tránh xung đột GPU)
                    async with manager.lock:
                        raw_probs, infer_latency = await asyncio.to_thread(
                            manager.forward_audio_slice, audio_slice
                        )

                    # Khóa kênh người nói và trích xuất danh sách active speakers
                    _, active_speakers = session.align_and_update_probs(raw_probs, offset_samples)

                    # Tính toán mốc thời gian thực của chunk vừa phân tích
                    t_start = (session.ring_buffer.last_emitted_sample_idx - session.chunk_samples) / DEFAULT_SAMPLE_RATE
                    t_end = session.ring_buffer.last_emitted_sample_idx / DEFAULT_SAMPLE_RATE

                    # 3. Phản hồi kết quả về Client
                    event_payload = {
                        "type": "diarization_event",
                        "session_id": session_id,
                        "chunk_index": session.chunk_count,
                        "time_start_sec": round(t_start, 3),
                        "time_end_sec": round(t_end, 3),
                        "active_speakers": active_speakers,
                        "num_active_speakers": len(active_speakers),
                        "unique_speakers_seen": list(session.active_speaker_history),
                        "latency_ms": round(infer_latency, 2),
                        "sla_passed": infer_latency <= 320.0
                    }
                    await websocket.send_json(event_payload)

            # 2. Nhận gói điều khiển dạng text/JSON
            elif "text" in message and message["text"]:
                try:
                    data = json.loads(message["text"])
                    action = data.get("action")
                    if action == "reset":
                        session.ring_buffer = AudioRingBuffer(sample_rate=DEFAULT_SAMPLE_RATE)
                        session.full_probs_so_far = None
                        session.chunk_count = 0
                        await websocket.send_json({"type": "reset_ack", "session_id": session_id})
                    elif action == "ping":
                        await websocket.send_json({"type": "pong", "server_time": time.time()})
                except Exception as e:
                    await websocket.send_json({"type": "error", "message": f"Invalid JSON payload: {e}"})

    except WebSocketDisconnect:
        print(f"[-] Client ngắt kết nối: {session_id}")
    except Exception as e:
        print(f"[!] Lỗi phiên WebSocket {session_id}: {e}")
    finally:
        active_sessions.pop(session_id, None)
        print(f"[*] Đã giải phóng tài nguyên phiên: {session_id} (Còn lại {len(active_sessions)} luồng)")


# ===============================================================================
# 7. GIAO DIỆN WEB DEMO TRỰC QUAN (LIVE MICROPHONE & AUDIO VISUALIZER)
# ===============================================================================

@app.get("/", response_class=HTMLResponse)
def get_demo_ui():
    """Giao diện web trực quan, hiện đại cho phép test Microphone hoặc Stream file trực tiếp."""
    html_content = """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nemotron-3 Real-Time Diarization (0.32s SLA)</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0e17;
      --card-bg: rgba(18, 24, 38, 0.75);
      --border-color: rgba(255, 255, 255, 0.08);
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-purple: #8b5cf6;
      --accent-orange: #f59e0b;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: radial-gradient(circle at 50% 0%, #151e33 0%, var(--bg-dark) 75%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 24px;
    }

    .container {
      max-width: 1100px;
      margin: 0 auto;
      width: 100%;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--border-color);
    }
    .brand h1 {
      font-size: 24px;
      font-weight: 700;
      background: linear-gradient(135deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand p { font-size: 13px; color: var(--text-muted); }

    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.3);
      color: var(--accent-green);
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      font-family: 'JetBrains Mono', monospace;
    }
    .status-dot {
      width: 8px; height: 8px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 8px currentColor;
    }

    /* Grid Layout */
    .dashboard-grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 20px;
    }

    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--border-color);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }

    .card-title {
      font-size: 14px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    /* Controls */
    .controls { display: flex; gap: 12px; }
    button {
      flex: 1;
      padding: 12px 20px;
      border-radius: 10px;
      border: none;
      font-family: 'Outfit', sans-serif;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.2s ease;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-blue), #2563eb);
      color: #fff;
      box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4);
    }
    .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
    .btn-danger {
      background: rgba(239, 68, 68, 0.2);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #f87171;
    }
    .btn-danger:hover { background: rgba(239, 68, 68, 0.3); }

    /* Visualizer */
    .visualizer-container {
      height: 80px;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 10px;
      overflow: hidden;
      margin-top: 16px;
      position: relative;
    }
    canvas { width: 100%; height: 100%; display: block; }

    /* Live Speakers List */
    .speaker-slots {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
      margin-top: 10px;
    }
    .speaker-card {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--border-color);
      border-radius: 10px;
      padding: 12px;
      display: flex;
      align-items: center;
      gap: 12px;
      transition: all 0.25s ease;
    }
    .speaker-card.active {
      border-color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.12);
      box-shadow: 0 0 16px rgba(6, 182, 212, 0.3);
      transform: scale(1.02);
    }
    .spk-avatar {
      width: 32px; height: 32px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 13px;
      background: rgba(255, 255, 255, 0.1);
    }
    .speaker-card.active .spk-avatar {
      background: var(--accent-cyan);
      color: #000;
    }
    .spk-info { flex: 1; }
    .spk-name { font-size: 13px; font-weight: 600; }
    .spk-prob { font-size: 11px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

    /* Metrics side */
    .metric-row {
      display: flex;
      justify-content: space-between;
      padding: 10px 0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      font-size: 13px;
    }
    .metric-val {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      color: var(--accent-cyan);
    }

    /* Logs view */
    .timeline-log {
      height: 280px;
      background: rgba(0, 0, 0, 0.4);
      border-radius: 10px;
      padding: 12px;
      overflow-y: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .log-item {
      padding: 6px 10px;
      border-radius: 6px;
      background: rgba(255, 255, 255, 0.03);
      border-left: 3px solid var(--accent-blue);
      display: flex;
      justify-content: space-between;
    }
    .log-item.speaker_0 { border-color: #3b82f6; }
    .log-item.speaker_1 { border-color: #10b981; }
    .log-item.speaker_2 { border-color: #f59e0b; }
    .log-item.speaker_3 { border-color: #ec4899; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <h1>NVIDIA Nemotron-3 Diarization</h1>
        <p>Kiến Trúc 1: WebSocket + Ring Buffer (0.32s SLA - Low Latency Serving)</p>
      </div>
      <div class="status-badge" id="wsStatus">
        <span class="status-dot"></span> SẴN SÀNG KẾT NỐI
      </div>
    </header>

    <div class="dashboard-grid">
      <!-- Cột Trái: Điều khiển & Người nói -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="card">
          <div class="card-title">🎙️ ĐIỀU KHIỂN THU ÂM TRỰC TIẾP</div>
          <div class="controls">
            <button class="btn-primary" id="btnStart" onclick="startRecording()">
              <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 02-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path></svg>
              Bắt Đầu Thu Âm
            </button>
            <button class="btn-danger" id="btnStop" onclick="stopRecording()" disabled>
              <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"></path></svg>
              Dừng Thu Âm
            </button>
          </div>

          <div class="visualizer-container">
            <canvas id="waveformCanvas"></canvas>
          </div>
        </div>

        <div class="card">
          <div class="card-title">
            <span>🔊 TRẠNG THÁI NGƯỜI NÓI (ACTIVE SPEAKERS)</span>
            <span style="font-size: 11px; color: var(--accent-cyan);" id="activeCount">0 người đang nói</span>
          </div>
          <div class="speaker-slots" id="speakerSlots">
            <!-- Sinh tự động bằng JS -->
          </div>
        </div>
      </div>

      <!-- Cột Phải: Thông số & Log -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        <div class="card">
          <div class="card-title">⚡ HIỆU NĂNG THỜI GIAN THỰC</div>
          <div class="metric-row">
            <span>Độ trễ tính toán chunk:</span>
            <span class="metric-val" id="valLatency">-- ms</span>
          </div>
          <div class="metric-row">
            <span>Tiêu chuẩn SLA:</span>
            <span class="metric-val" style="color: #10b981;">&le; 320 ms</span>
          </div>
          <div class="metric-row">
            <span>Tổng chunks đã xử lý:</span>
            <span class="metric-val" id="valChunks">0</span>
          </div>
          <div class="metric-row">
            <span>Thời lượng âm thanh:</span>
            <span class="metric-val" id="valAudioTime">0.00 s</span>
          </div>
        </div>

        <div class="card" style="flex: 1;">
          <div class="card-title">📋 NHẬT KÝ SỰ KIỆN (REAL-TIME TIMELINE)</div>
          <div class="timeline-log" id="timelineLog">
            <div style="color: var(--text-muted); text-align: center; margin-top: 30px;">
              Bấm 'Bắt Đầu Thu Âm' để nhận diện người nói trực tiếp...
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let socket = null;
    let audioContext = null;
    let micStream = null;
    let processorNode = null;
    let isRecording = false;

    // Khởi tạo các slot người nói từ 0 đến 3
    const speakerSlotsEl = document.getElementById("speakerSlots");
    for (let i = 0; i < 4; i++) {
      const card = document.createElement("div");
      card.className = "speaker-card";
      card.id = `spkCard_${i}`;
      card.innerHTML = `
        <div class="spk-avatar">S${i}</div>
        <div class="spk-info">
          <div class="spk-name">SPEAKER_${String(i).padStart(2, '0')}</div>
          <div class="spk-prob" id="spkProb_${i}">Im lặng</div>
        </div>
      `;
      speakerSlotsEl.appendChild(card);
    }

    // Kết nối WebSocket
    function connectWebSocket() {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${proto}//${window.location.host}/ws/diarize`;
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        document.getElementById("wsStatus").innerHTML = '<span class="status-dot"></span> ĐÃ KẾT NỐI WEBSOCKET';
        document.getElementById("wsStatus").style.color = 'var(--accent-green)';
      };

      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'diarization_event') {
          handleDiarizationEvent(data);
        }
      };

      socket.onclose = () => {
        document.getElementById("wsStatus").innerHTML = '<span class="status-dot" style="background:#ef4444;"></span> MẤT KẾT NỐI';
        document.getElementById("wsStatus").style.color = '#ef4444';
        if (isRecording) stopRecording();
      };
    }

    function handleDiarizationEvent(event) {
      document.getElementById("valLatency").innerText = `${event.latency_ms} ms`;
      document.getElementById("valChunks").innerText = event.chunk_index;
      document.getElementById("valAudioTime").innerText = `${event.time_end_sec} s`;
      document.getElementById("activeCount").innerText = `${event.num_active_speakers} người đang nói`;

      // Cập nhật card trạng thái
      const activeIds = new Set(event.active_speakers.map(s => s.channel_index));
      for (let i = 0; i < 4; i++) {
        const card = document.getElementById(`spkCard_${i}`);
        const probEl = document.getElementById(`spkProb_${i}`);
        if (activeIds.has(i)) {
          card.classList.add("active");
          const spkData = event.active_speakers.find(s => s.channel_index === i);
          probEl.innerText = `Nói (Conf: ${(spkData.confidence * 100).toFixed(1)}%)`;
        } else {
          card.classList.remove("active");
          probEl.innerText = "Im lặng";
        }
      }

      // Thêm dòng log nếu có người nói
      if (event.active_speakers.length > 0) {
        const logContainer = document.getElementById("timelineLog");
        const spkNames = event.active_speakers.map(s => s.speaker_id).join(", ");
        const firstChan = event.active_speakers[0].channel_index;

        const row = document.createElement("div");
        row.className = `log-item speaker_${firstChan % 4}`;
        row.innerHTML = `
          <span>[${event.time_start_sec.toFixed(2)}s - ${event.time_end_sec.toFixed(2)}s] <strong>${spkNames}</strong></span>
          <span style="color:var(--text-muted);">${event.latency_ms}ms</span>
        `;
        logContainer.appendChild(row);
        logContainer.scrollTop = logContainer.scrollHeight;
      }
    }

    // Bắt đầu thu âm trực tiếp từ Microphone
    async function startRecording() {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        await new Promise(r => setTimeout(r, 500));
      }

      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(micStream);

        // Buffer size 4096 samples (~256ms)
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);
        source.connect(processorNode);
        processorNode.connect(audioContext.destination);

        processorNode.onaudioprocess = (e) => {
          if (!isRecording || socket.readyState !== WebSocket.OPEN) return;
          const inputData = e.inputBuffer.getChannelData(0);

          // Chuyển Float32 sang 16-bit PCM bytes
          const pcm16 = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            const s = Math.max(-1, Math.min(1, inputData[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          socket.send(pcm16.buffer);
        };

        isRecording = true;
        document.getElementById("btnStart").disabled = true;
        document.getElementById("btnStop").disabled = false;
        document.getElementById("timelineLog").innerHTML = "";
      } catch (err) {
        alert("Không thể truy cập Microphone: " + err);
      }
    }

    function stopRecording() {
      isRecording = false;
      if (processorNode) processorNode.disconnect();
      if (micStream) micStream.getTracks().forEach(t => t.stop());
      if (audioContext) audioContext.close();

      document.getElementById("btnStart").disabled = false;
      document.getElementById("btnStop").disabled = true;
    }

    window.onload = connectWebSocket;
  </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# ===============================================================================
# 8. ĐIỂM VÀO CLI
# ===============================================================================

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description="Khởi chạy WebSocket Streaming Server cho Nemotron-3 Diarization")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Địa chỉ Host")
    parser.add_argument("--port", type=int, default=8000, help="Cổng Port")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn checkpoint model SWA")
    args = parser.parse_args()

    if args.model:
        ModelManager.get_instance().load_model(args.model)

    print(f"\n[*] Đang khởi động Server tại: http://{args.host}:{args.port}")
    print(f"[*] WebSocket URL: ws://{args.host}:{args.port}/ws/diarize")
    print(f"[*] Giao diện Demo UI: http://localhost:{args.port}/\n")

    uvicorn.run(app, host=args.host, port=args.port)
