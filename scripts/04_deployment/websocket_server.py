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


from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


def parse_rttm_text(rttm_text: str, uri: str = "eval_session") -> Annotation:
    """Đọc chuỗi RTTM chuẩn NIST thành pyannote.core.Annotation."""
    annotation = Annotation(uri=uri)
    for line in rttm_text.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("SPEAKER"):
            continue
        parts = line.split()
        if len(parts) >= 8:
            try:
                start = float(parts[3])
                duration = float(parts[4])
                spk = parts[7]
                if duration > 0.001:
                    annotation[Segment(start, start + duration)] = spk
            except Exception:
                continue
    return annotation


def compute_pair_metrics(ref_rttm_str: str, hyp_rttm_str: str) -> Dict[str, Any]:
    """Tính toán chi tiết các chỉ số NIST DER (c0 & c0.25), Miss, FA, Conf, JER và SCA."""
    uri = "eval_session"
    ref = parse_rttm_text(ref_rttm_str, uri=uri)
    hyp = parse_rttm_text(hyp_rttm_str, uri=uri)

    if len(ref) == 0:
        raise ValueError("Ground Truth RTTM không chứa đoạn phát biểu hợp lệ!")
    if len(hyp) == 0:
        raise ValueError("Kết quả RTTM dự đoán rỗng (không phát hiện người nói)!")

    der_c0_metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    der_c025_metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    jer_metric = JaccardErrorRate(collar=0.0, skip_overlap=False)

    der_c0 = der_c0_metric(ref, hyp) * 100.0
    der_c025 = der_c025_metric(ref, hyp) * 100.0
    jer = jer_metric(ref, hyp) * 100.0

    acc_c0 = der_c0_metric.accumulated_
    tot_c0 = acc_c0.get("total", 0.0)
    miss_c0 = (acc_c0.get("missed detection", 0.0) / max(1e-6, tot_c0)) * 100.0
    fa_c0 = (acc_c0.get("false alarm", 0.0) / max(1e-6, tot_c0)) * 100.0
    conf_c0 = (acc_c0.get("confusion", 0.0) / max(1e-6, tot_c0)) * 100.0

    ref_speakers = len(ref.labels())
    hyp_speakers = len(hyp.labels())

    return {
        "der_collar0": round(der_c0, 2),
        "der_collar025": round(der_c025, 2),
        "miss_rate_c0": round(miss_c0, 2),
        "false_alarm_c0": round(fa_c0, 2),
        "confusion_c0": round(conf_c0, 2),
        "jer": round(jer, 2),
        "ref_speakers": ref_speakers,
        "hyp_speakers": hyp_speakers,
        "speaker_count_match": ref_speakers == hyp_speakers,
        "speaker_count_diff": abs(ref_speakers - hyp_speakers),
        "ref_speech_duration_sec": round(tot_c0, 2)
    }


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


@app.post("/api/evaluate_rttm")
async def evaluate_rttm_api(request: Request):
    """API đối chiếu kết quả dự đoán với Ground Truth RTTM và trả về bộ metrics chuẩn NIST."""
    try:
        body = await request.json()
        ref_text = body.get("reference_rttm", "")
        hyp_text = body.get("hypothesis_rttm", "")

        if not ref_text.strip():
            return JSONResponse({"status": "error", "message": "Chưa cung cấp Ground Truth RTTM!"}, status_code=400)
        if not hyp_text.strip():
            return JSONResponse({"status": "error", "message": "Chưa có dữ liệu RTTM dự đoán!"}, status_code=400)

        metrics = compute_pair_metrics(ref_text, hyp_text)
        return JSONResponse({"status": "success", "metrics": metrics})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


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
    """Giao diện web trực quan, hiện đại: Hỗ trợ Live Mic / File Stream, Xuất RTTM tức thì & Đo Metrics với Ground Truth."""
    html_content = """
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nemotron-3 Real-Time Diarization & Benchmark Hub</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-dark: #0a0e17;
      --card-bg: rgba(18, 24, 38, 0.78);
      --card-border: rgba(255, 255, 255, 0.08);
      --accent-blue: #3b82f6;
      --accent-cyan: #06b6d4;
      --accent-green: #10b981;
      --accent-purple: #8b5cf6;
      --accent-orange: #f59e0b;
      --accent-red: #ef4444;
      --text-main: #f3f4f6;
      --text-muted: #9ca3af;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Outfit', sans-serif;
      background: radial-gradient(circle at 50% 0%, #151e33 0%, var(--bg-dark) 80%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      padding: 20px;
    }

    .container {
      max-width: 1240px;
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
      border-bottom: 1px solid var(--card-border);
    }
    .brand h1 {
      font-size: 22px;
      font-weight: 800;
      background: linear-gradient(135deg, #60a5fa, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand p { font-size: 13px; color: var(--text-muted); }

    .status-badge {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(16, 185, 129, 0.12);
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

    /* Grid layout */
    .grid-main {
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 20px;
    }

    .card {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }

    /* Tabs */
    .tab-group {
      display: flex;
      background: rgba(0, 0, 0, 0.3);
      border-radius: 10px;
      padding: 3px;
      gap: 4px;
    }
    .tab-btn {
      flex: 1;
      padding: 8px 12px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font-family: 'Outfit', sans-serif;
      font-size: 13px;
      font-weight: 600;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.2s;
    }
    .tab-btn.active {
      background: var(--accent-blue);
      color: #fff;
      box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
    }

    /* Controls */
    .controls-row {
      display: flex;
      gap: 12px;
    }
    button.btn {
      padding: 12px 18px;
      border-radius: 10px;
      border: none;
      font-family: 'Outfit', sans-serif;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.2s ease;
    }
    button.btn:disabled { opacity: 0.4; cursor: not-allowed; }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent-blue), #2563eb);
      color: #fff;
      box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
    }
    .btn-primary:hover:not(:disabled) { transform: translateY(-1px); }
    .btn-danger {
      background: rgba(239, 68, 68, 0.18);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #f87171;
    }
    .btn-danger:hover:not(:disabled) { background: rgba(239, 68, 68, 0.28); }
    .btn-secondary {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--card-border);
      color: var(--text-main);
    }
    .btn-secondary:hover:not(:disabled) { background: rgba(255, 255, 255, 0.14); }

    /* Ground Truth Upload Box */
    .gt-box {
      border: 1px dashed rgba(255, 255, 255, 0.18);
      border-radius: 12px;
      padding: 14px;
      background: rgba(0, 0, 0, 0.25);
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .file-input-wrapper {
      display: flex;
      align-items: center;
      gap: 10px;
    }
    input[type="file"] { display: none; }
    .custom-file-btn {
      padding: 7px 14px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid var(--card-border);
      color: var(--text-main);
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
    }
    .custom-file-btn:hover { background: rgba(255, 255, 255, 0.16); }

    /* Active Speakers */
    .speaker-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }
    .spk-badge {
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 10px;
      display: flex;
      align-items: center;
      gap: 10px;
      transition: all 0.2s ease;
    }
    .spk-badge.active {
      border-color: var(--accent-cyan);
      background: rgba(6, 182, 212, 0.14);
      box-shadow: 0 0 14px rgba(6, 182, 212, 0.35);
      transform: translateY(-2px);
    }
    .spk-avatar {
      width: 32px; height: 32px;
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.08);
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 13px;
    }
    .spk-badge.active .spk-avatar {
      background: var(--accent-cyan);
      color: #000;
    }
    .spk-title { font-size: 13px; font-weight: 600; }
    .spk-status { font-size: 11px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

    /* Waveform Canvas */
    .visualizer-box {
      height: 60px;
      background: rgba(0, 0, 0, 0.4);
      border-radius: 10px;
      overflow: hidden;
    }
    canvas { width: 100%; height: 100%; display: block; }

    /* Results section after stop */
    .results-card {
      border-color: rgba(96, 165, 250, 0.3);
      background: linear-gradient(180deg, rgba(30, 41, 59, 0.7) 0%, rgba(18, 24, 38, 0.85) 100%);
    }

    /* Scorecard metrics */
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 12px;
    }
    .metric-item {
      background: rgba(0, 0, 0, 0.35);
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .metric-item.winner {
      border-color: rgba(16, 185, 129, 0.4);
      background: rgba(16, 185, 129, 0.08);
    }
    .metric-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }
    .metric-value {
      font-size: 20px;
      font-weight: 700;
      font-family: 'JetBrains Mono', monospace;
      color: var(--accent-cyan);
    }
    .metric-sub { font-size: 11px; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }

    /* RTTM Code Box */
    .rttm-box {
      background: #06090f;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px;
      font-family: 'JetBrains Mono', monospace;
      font-size: 12px;
      line-height: 1.5;
      color: #93c5fd;
      max-height: 220px;
      overflow-y: auto;
      white-space: pre;
    }

    /* Timeline Log */
    .timeline-log {
      height: 260px;
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
    .log-item.spk_0 { border-color: #3b82f6; }
    .log-item.spk_1 { border-color: #10b981; }
    .log-item.spk_2 { border-color: #f59e0b; }
    .log-item.spk_3 { border-color: #ec4899; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="brand">
        <h1>NVIDIA Nemotron-3 Diarization</h1>
        <p>Production Streaming Hub (0.32s SLA) & NIST RTTM Evaluation Playground</p>
      </div>
      <div class="status-badge" id="wsStatus">
        <span class="status-dot"></span> SẴN SÀNG KẾT NỐI
      </div>
    </header>

    <div class="grid-main">
      <!-- Cột Trái: Điều Khiển + Ground Truth + Kết Quả RTTM -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        
        <!-- CARD 1: ĐIỀU KHIỂN STREAMING -->
        <div class="card">
          <div class="card-header">
            <span>🎙️ NGUỒN ÂM THANH STREAMING</span>
            <span style="font-size: 11px; color: var(--accent-cyan);" id="streamTimeLabel">0.00s</span>
          </div>

          <div class="tab-group">
            <button class="tab-btn active" id="tabMic" onclick="switchInputMode('mic')">Microphone Trực Tiếp</button>
            <button class="tab-btn" id="tabFile" onclick="switchInputMode('file')">Stream File WAV / MP3</button>
          </div>

          <div id="fileUploadRow" style="display: none;">
            <div class="file-input-wrapper">
              <label class="custom-file-btn" for="audioFileInput">📁 Chọn File Âm Thanh...</label>
              <input type="file" id="audioFileInput" accept="audio/*" onchange="handleAudioFileSelect(event)">
              <span id="audioFileName" style="font-size: 12px; color: var(--text-muted);">Chưa chọn file âm thanh</span>
            </div>
          </div>

          <div class="controls-row">
            <button class="btn btn-primary" id="btnStart" onclick="startStreaming()">
              <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              Bắt Đầu Thu / Stream
            </button>
            <button class="btn btn-danger" id="btnStop" onclick="stopStreaming()" disabled>
              <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"></path></svg>
              Dừng Thu Âm & Xuất RTTM
            </button>
            <button class="btn btn-secondary" onclick="resetAll()">Làm Mới</button>
          </div>

          <div class="visualizer-box">
            <canvas id="waveformCanvas"></canvas>
          </div>
        </div>

        <!-- CARD 2: GROUND TRUTH RTTM (TẢI LÊN ĐỂ ĐỐI CHIẾU) -->
        <div class="card">
          <div class="card-header">
            <span>📁 GROUND TRUTH (NHÃN RTTM GỐC ĐỂ CHẤM ĐIỂM METRICS)</span>
            <span id="gtBadge" style="font-size: 11px; color: var(--accent-orange);">Chưa có nhãn gốc</span>
          </div>

          <div class="gt-box">
            <div class="file-input-wrapper">
              <label class="custom-file-btn" for="rttmFileInput">Tải Lên File .rttm</label>
              <input type="file" id="rttmFileInput" accept=".rttm,.txt" onchange="handleRttmFileSelect(event)">
              <span id="rttmFileName" style="font-size: 12px; color: var(--text-muted);">Hoặc dán nội dung RTTM trực tiếp vào ô bên dưới</span>
            </div>
            <textarea id="gtRttmText" rows="3" placeholder="SPEAKER audio_name 1 0.00 4.52 <NA> <NA> SPEAKER_01 <NA> <NA>&#10;SPEAKER audio_name 1 5.10 3.20 <NA> <NA> SPEAKER_02 <NA> <NA>" style="background: rgba(0,0,0,0.3); border: 1px solid var(--card-border); border-radius: 8px; color: var(--text-main); font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 8px; resize: vertical;" oninput="updateGtStats()"></textarea>
          </div>
        </div>

        <!-- CARD 3: BẢNG KẾT QUẢ ĐO ĐẠC METRICS & HYPOTHESIS RTTM (HIỆN KHI DỪNG THU ÂM) -->
        <div class="card results-card" id="resultsSection" style="display: none;">
          <div class="card-header">
            <span>🏆 KẾT QUẢ NGHIỆM THU RTTM & ĐỐI CHIẾU METRICS</span>
            <div style="display: flex; gap: 8px;">
              <button class="custom-file-btn" onclick="copyRttmToClipboard()">📋 Sao Chép RTTM</button>
              <button class="custom-file-btn" onclick="downloadRttmFile()">💾 Tải Về .rttm</button>
            </div>
          </div>

          <!-- Scorecard khi có Ground Truth -->
          <div id="metricsScorecard" style="display: none; flex-direction: column; gap: 10px;">
            <div style="font-size: 13px; font-weight: 600; color: #a78bfa;">📊 BẢNG ĐIỂM CHUẨN NIST (SO SÁNH VỚI GROUND TRUTH):</div>
            <div class="metric-grid">
              <div class="metric-item winner">
                <span class="metric-label">DER Collar 0.0s (Nghiêm ngặt)</span>
                <span class="metric-value" id="valDerC0">--%</span>
                <span class="metric-sub" id="valDerC0Sub">Miss: --% | FA: --% | Conf: --%</span>
              </div>
              <div class="metric-item winner">
                <span class="metric-label">DER Collar 0.25s (Chuẩn NIST)</span>
                <span class="metric-value" id="valDerC025" style="color: #10b981;">--%</span>
                <span class="metric-sub">Khoảng dung sai tiêu chuẩn</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">Jaccard Error Rate (JER)</span>
                <span class="metric-value" id="valJer">--%</span>
                <span class="metric-sub">Độ phủ phân đoạn người nói</span>
              </div>
              <div class="metric-item">
                <span class="metric-label">Độ Chính Xác Đếm Người (SCA)</span>
                <span class="metric-value" id="valSca">--</span>
                <span class="metric-sub" id="valScaSub">Ref: - | Hyp: -</span>
              </div>
            </div>
          </div>

          <div id="noGtNotice" style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.3); padding: 10px 14px; border-radius: 8px; font-size: 12px; color: var(--text-muted);">
            <span>Chưa có Ground Truth RTTM để đối chiếu metrics. Hãy tải file RTTM lên và bấm nút tính điểm:</span>
            <button class="custom-file-btn" onclick="calculateMetricsNow()">🎯 Tính Toán Metrics Ngay</button>
          </div>

          <!-- RTTM Output Preview -->
          <div style="display: flex; flex-direction: column; gap: 6px;">
            <span style="font-size: 12px; font-weight: 600; color: var(--text-muted);">📜 FILE RTTM XUẤT XƯỞNG CHUẨN NIST (HYPOTHESIS OUTPUT):</span>
            <div class="rttm-box" id="rttmOutputBox">Chưa có kết quả RTTM.</div>
          </div>
        </div>

      </div>

      <!-- Cột Phải: Trạng Thái Người Nói & Timeline Event -->
      <div style="display: flex; flex-direction: column; gap: 20px;">
        
        <!-- CARD 4: TRẠNG THÁI NGƯỜI NÓI HIỆN TẠI -->
        <div class="card">
          <div class="card-header">
            <span>🔊 TRẠNG THÁI NGƯỜI NÓI (LIVE)</span>
            <span style="font-size: 11px; color: var(--accent-cyan);" id="activeSpeakerCountLabel">0 người</span>
          </div>

          <div class="speaker-grid" id="speakerSlots">
            <!-- 4 Speaker Badges -->
          </div>

          <div style="display: flex; justify-content: space-between; padding-top: 8px; border-top: 1px solid var(--card-border); font-size: 12px;">
            <span>Độ trễ chunk hiện tại:</span>
            <span style="font-family:'JetBrains Mono'; font-weight:700; color:var(--accent-cyan);" id="valLatency">-- ms</span>
          </div>
        </div>

        <!-- CARD 5: TIMELINE LOGS -->
        <div class="card" style="flex: 1;">
          <div class="card-header">
            <span>📋 NHẬT KÝ SỰ KIỆN (REAL-TIME TIMELINE)</span>
            <span style="font-size: 11px; color: var(--text-muted);" id="chunkCountLabel">0 chunks</span>
          </div>

          <div class="timeline-log" id="timelineLog">
            <div style="color: var(--text-muted); text-align: center; margin-top: 40px;">
              Bấm 'Bắt Đầu Thu' để xem sự kiện phân đoạn...
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
    let isStreaming = false;
    let inputMode = 'mic'; // 'mic' | 'file'
    let selectedAudioFile = null;

    // Bộ nhớ lưu trữ tất cả các chunks đã nhận được trong phiên
    let recordedChunks = [];
    let currentHypothesisRttm = "";

    // 1. Khởi tạo 4 Speaker slots
    const speakerSlotsEl = document.getElementById("speakerSlots");
    for (let i = 0; i < 4; i++) {
      const badge = document.createElement("div");
      badge.className = "spk-badge";
      badge.id = `spkBadge_${i}`;
      badge.innerHTML = `
        <div class="spk-avatar">S${i}</div>
        <div>
          <div class="spk-title">SPK_${String(i).padStart(2, '0')}</div>
          <div class="spk-status" id="spkStatus_${i}">Im lặng</div>
        </div>
      `;
      speakerSlotsEl.appendChild(badge);
    }

    // 2. Chuyển đổi chế độ Mic / File
    function switchInputMode(mode) {
      if (isStreaming) return;
      inputMode = mode;
      document.getElementById("tabMic").classList.toggle("active", mode === 'mic');
      document.getElementById("tabFile").classList.toggle("active", mode === 'file');
      document.getElementById("fileUploadRow").style.display = (mode === 'file') ? "block" : "none";
    }

    function handleAudioFileSelect(e) {
      const file = e.target.files[0];
      if (file) {
        selectedAudioFile = file;
        document.getElementById("audioFileName").innerText = `${file.name} (${(file.size/1024/1024).toFixed(2)} MB)`;
      }
    }

    function handleRttmFileSelect(e) {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          document.getElementById("gtRttmText").value = event.target.result;
          document.getElementById("rttmFileName").innerText = file.name;
          updateGtStats();
        };
        reader.readAsText(file);
      }
    }

    function updateGtStats() {
      const txt = document.getElementById("gtRttmText").value.trim();
      if (!txt) {
        document.getElementById("gtBadge").innerText = "Chưa có nhãn gốc";
        document.getElementById("gtBadge").style.color = "var(--accent-orange)";
        return;
      }
      const lines = txt.split('\\n').filter(l => l.trim().startsWith("SPEAKER"));
      const spks = new Set();
      lines.forEach(l => {
        const p = l.trim().split(/\\s+/);
        if (p.length >= 8) spks.add(p[7]);
      });
      document.getElementById("gtBadge").innerText = `Đã nạp: ${spks.size} người nói (${lines.length} đoạn)`;
      document.getElementById("gtBadge").style.color = "var(--accent-green)";
    }

    // 3. Kết nối WebSocket
    function connectWebSocket(onReady) {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${proto}//${window.location.host}/ws/diarize`;
      socket = new WebSocket(wsUrl);

      socket.onopen = () => {
        document.getElementById("wsStatus").innerHTML = '<span class="status-dot"></span> ĐÃ KẾT NỐI WEBSOCKET';
        document.getElementById("wsStatus").style.color = 'var(--accent-green)';
        if (onReady) onReady();
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
        if (isStreaming) stopStreaming();
      };
    }

    // 4. Xử lý sự kiện nhận từ server
    function handleDiarizationEvent(event) {
      recordedChunks.push(event);

      document.getElementById("valLatency").innerText = `${event.latency_ms} ms`;
      document.getElementById("streamTimeLabel").innerText = `${event.time_end_sec.toFixed(2)}s`;
      document.getElementById("chunkCountLabel").innerText = `${event.chunk_index} chunks`;
      document.getElementById("activeSpeakerCountLabel").innerText = `${event.num_active_speakers} người`;

      // Cập nhật card trạng thái loa
      const activeChans = new Set(event.active_speakers.map(s => s.channel_index));
      for (let i = 0; i < 4; i++) {
        const badge = document.getElementById(`spkBadge_${i}`);
        const statusEl = document.getElementById(`spkStatus_${i}`);
        if (activeChans.has(i)) {
          badge.classList.add("active");
          const spkInfo = event.active_speakers.find(s => s.channel_index === i);
          statusEl.innerText = `Nói (${(spkInfo.confidence * 100).toFixed(0)}%)`;
        } else {
          badge.classList.remove("active");
          statusEl.innerText = "Im lặng";
        }
      }

      // Thêm log timeline
      if (event.active_speakers.length > 0) {
        const logContainer = document.getElementById("timelineLog");
        const spkNames = event.active_speakers.map(s => s.speaker_id).join(", ");
        const firstChan = event.active_speakers[0].channel_index;

        const row = document.createElement("div");
        row.className = `log-item spk_${firstChan % 4}`;
        row.innerHTML = `
          <span>[${event.time_start_sec.toFixed(2)}s - ${event.time_end_sec.toFixed(2)}s] <strong>${spkNames}</strong></span>
          <span style="color:var(--text-muted);">${event.latency_ms}ms</span>
        `;
        logContainer.appendChild(row);
        logContainer.scrollTop = logContainer.scrollHeight;
      }
    }

    // 5. Bắt đầu Stream
    function startStreaming() {
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        connectWebSocket(() => startStreamingAction());
      } else {
        startStreamingAction();
      }
    }

    async function startStreamingAction() {
      recordedChunks = [];
      document.getElementById("timelineLog").innerHTML = "";
      document.getElementById("resultsSection").style.display = "none";
      document.getElementById("metricsScorecard").style.display = "none";

      if (inputMode === 'mic') {
        startMicCapture();
      } else {
        if (!selectedAudioFile) {
          alert("Vui lòng chọn 1 file âm thanh WAV hoặc MP3 trước khi bắt đầu!");
          return;
        }
        startFileStreaming(selectedAudioFile);
      }
    }

    async function startMicCapture() {
      try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const source = audioContext.createMediaStreamSource(micStream);

        processorNode = audioContext.createScriptProcessor(4096, 1, 1);
        source.connect(processorNode);
        processorNode.connect(audioContext.destination);

        processorNode.onaudioprocess = (e) => {
          if (!isStreaming || socket.readyState !== WebSocket.OPEN) return;
          const input = e.inputBuffer.getChannelData(0);
          const pcm16 = new Int16Array(input.length);
          for (let i = 0; i < input.length; i++) {
            const s = Math.max(-1, Math.min(1, input[i]));
            pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
          }
          socket.send(pcm16.buffer);
        };

        isStreaming = true;
        document.getElementById("btnStart").disabled = true;
        document.getElementById("btnStop").disabled = false;
      } catch (err) {
        alert("Lỗi truy cập Microphone: " + err);
      }
    }

    async function startFileStreaming(file) {
      isStreaming = true;
      document.getElementById("btnStart").disabled = true;
      document.getElementById("btnStop").disabled = false;

      const arrayBuffer = await file.arrayBuffer();
      const tempAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const decodedBuffer = await tempAudioCtx.decodeAudioData(arrayBuffer);

      // Resample sang 16kHz
      const offlineCtx = new OfflineAudioContext(1, Math.ceil(decodedBuffer.duration * 16000), 16000);
      const offlineSource = offlineCtx.createBufferSource();
      offlineSource.buffer = decodedBuffer;
      offlineSource.connect(offlineCtx.destination);
      offlineSource.start();
      const resampledBuffer = await offlineCtx.startRendering();

      const floatData = resampledBuffer.getChannelData(0);
      const chunkSamples = 3840; // 240ms @ 16kHz

      let offset = 0;
      const streamLoop = setInterval(() => {
        if (!isStreaming || offset >= floatData.length || socket.readyState !== WebSocket.OPEN) {
          clearInterval(streamLoop);
          if (isStreaming) stopStreaming();
          return;
        }

        const slice = floatData.subarray(offset, offset + chunkSamples);
        const pcm16 = new Int16Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
          const s = Math.max(-1, Math.min(1, slice[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }
        socket.send(pcm16.buffer);
        offset += chunkSamples;
      }, 240);
    }

    // 6. Dừng Thu Âm & Tạo File RTTM Tức Thì
    function stopStreaming() {
      isStreaming = false;
      if (processorNode) processorNode.disconnect();
      if (micStream) micStream.getTracks().forEach(t => t.stop());
      if (audioContext) audioContext.close();

      document.getElementById("btnStart").disabled = false;
      document.getElementById("btnStop").disabled = true;

      // Sinh file RTTM từ các chunks đã lưu
      generateAndDisplayRttm();
    }

    function generateAndDisplayRttm() {
      if (recordedChunks.length === 0) {
        alert("Chưa ghi nhận được âm thanh nào để xuất RTTM!");
        return;
      }

      // Thuật toán gộp các chunk liên tục của cùng speaker
      const speakerIntervals = {};
      recordedChunks.forEach(chunk => {
        chunk.active_speakers.forEach(spk => {
          const sid = spk.speaker_id;
          if (!speakerIntervals[sid]) speakerIntervals[sid] = [];
          speakerIntervals[sid].push({ start: chunk.time_start_sec, end: chunk.time_end_sec });
        });
      });

      const rttmLines = [];
      const uri = selectedAudioFile ? selectedAudioFile.name.replace(/\\.[^/.]+$/, "") : "live_stream_session";

      Object.keys(speakerIntervals).sort().forEach(spkId => {
        const intervals = speakerIntervals[spkId].sort((a, b) => a.start - b.start);
        let curStart = null;
        let curEnd = null;

        intervals.forEach(it => {
          if (curStart === null) {
            curStart = it.start;
            curEnd = it.end;
          } else {
            // Nối khoảng ngắt nhỏ hơn 0.15s (min_gap_off)
            if (it.start - curEnd <= 0.16) {
              curEnd = it.end;
            } else {
              const dur = curEnd - curStart;
              if (dur >= 0.05) {
                rttmLines.push(`SPEAKER ${uri} 1 ${curStart.toFixed(2)} ${dur.toFixed(2)} <NA> <NA> ${spkId} <NA> <NA>`);
              }
              curStart = it.start;
              curEnd = it.end;
            }
          }
        });

        if (curStart !== null) {
          const dur = curEnd - curStart;
          if (dur >= 0.05) {
            rttmLines.push(`SPEAKER ${uri} 1 ${curStart.toFixed(2)} ${dur.toFixed(2)} <NA> <NA> ${spkId} <NA> <NA>`);
          }
        }
      });

      // Sắp xếp theo mốc thời gian bắt đầu
      rttmLines.sort((a, b) => parseFloat(a.split(/\\s+/)[3]) - parseFloat(b.split(/\\s+/)[3]));

      currentHypothesisRttm = rttmLines.join("\\n");
      document.getElementById("rttmOutputBox").innerText = currentHypothesisRttm || "Không có đoạn phát biểu nào được ghi nhận.";
      document.getElementById("resultsSection").style.display = "flex";

      // Nếu có Ground Truth RTTM, tự động tính toán metrics luôn
      const gtText = document.getElementById("gtRttmText").value.trim();
      if (gtText && currentHypothesisRttm) {
        calculateMetricsNow();
      } else {
        document.getElementById("noGtNotice").style.display = "flex";
        document.getElementById("metricsScorecard").style.display = "none";
      }
    }

    // 7. Gọi API Chấm Điểm Metrics NIST với Ground Truth
    async function calculateMetricsNow() {
      const gtText = document.getElementById("gtRttmText").value.trim();
      if (!gtText) {
        alert("Vui lòng tải lên hoặc dán nội dung Ground Truth RTTM trước!");
        return;
      }
      if (!currentHypothesisRttm) {
        alert("Chưa có kết quả RTTM dự đoán!");
        return;
      }

      try {
        const resp = await fetch("/api/evaluate_rttm", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            reference_rttm: gtText,
            hypothesis_rttm: currentHypothesisRttm
          })
        });
        const res = await resp.json();

        if (res.status === "success") {
          const m = res.metrics;
          document.getElementById("valDerC0").innerText = `${m.der_collar0.toFixed(2)}%`;
          document.getElementById("valDerC0Sub").innerText = `Miss: ${m.miss_rate_c0}% | FA: ${m.false_alarm_c0}% | Conf: ${m.confusion_c0}%`;
          document.getElementById("valDerC025").innerText = `${m.der_collar025.toFixed(2)}%`;
          document.getElementById("valJer").innerText = `${m.jer.toFixed(2)}%`;
          document.getElementById("valSca").innerText = m.speaker_count_match ? "✅ CHÍNH XÁC" : "⚠️ SAI LỆCH";
          document.getElementById("valScaSub").innerText = `Ref: ${m.ref_speakers} spks | Hyp: ${m.hyp_speakers} spks`;

          document.getElementById("metricsScorecard").style.display = "flex";
          document.getElementById("noGtNotice").style.display = "none";
        } else {
          alert("Lỗi tính toán metrics: " + res.message);
        }
      } catch (err) {
        alert("Không thể kết nối đến API tính toán metrics: " + err);
      }
    }

    function copyRttmToClipboard() {
      if (!currentHypothesisRttm) return;
      navigator.clipboard.writeText(currentHypothesisRttm).then(() => {
        alert("Đã sao chép nội dung RTTM vào bộ nhớ đệm!");
      });
    }

    function downloadRttmFile() {
      if (!currentHypothesisRttm) return;
      const blob = new Blob([currentHypothesisRttm], { type: "text/plain;charset=utf-8" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "hypothesis_output.rttm";
      a.click();
    }

    function resetAll() {
      if (isStreaming) stopStreaming();
      recordedChunks = [];
      currentHypothesisRttm = "";
      document.getElementById("timelineLog").innerHTML = "";
      document.getElementById("rttmOutputBox").innerText = "Chưa có kết quả RTTM.";
      document.getElementById("resultsSection").style.display = "none";
      document.getElementById("metricsScorecard").style.display = "none";
      document.getElementById("streamTimeLabel").innerText = "0.00s";
      document.getElementById("chunkCountLabel").innerText = "0 chunks";
      document.getElementById("valLatency").innerText = "-- ms";
      document.getElementById("activeSpeakerCountLabel").innerText = "0 người";
      for (let i = 0; i < 4; i++) {
        document.getElementById(`spkBadge_${i}`).classList.remove("active");
        document.getElementById(`spkStatus_${i}`).innerText = "Im lặng";
      }
    }

    window.onload = () => connectWebSocket();
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
