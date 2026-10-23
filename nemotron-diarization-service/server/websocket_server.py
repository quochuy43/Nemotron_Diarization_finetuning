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
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForAudioFrameClassification

# Cấu hình đường dẫn nội bộ Service
SERVER_DIR = Path(__file__).resolve().parent
ROOT_DIR = SERVER_DIR.parent


def resolve_file_path(path_str: str) -> str:
    """Tự động chuẩn hóa và tìm kiếm file trên hệ thống."""
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.exists():
        return str(p.resolve())
    for base in [ROOT_DIR, SERVER_DIR, ROOT_DIR / "checkpoints"]:
        candidate = base / path_str
        if candidate.exists():
            return str(candidate.resolve())
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

DIARIZATION_API_KEY = os.environ.get("DIARIZATION_API_KEY", "")
MAX_CONCURRENT_SESSIONS = int(os.environ.get("MAX_CONCURRENT_SESSIONS", "50"))
MAX_WS_FRAME_BYTES = 1024 * 1024   # 1MB max per WebSocket binary frame


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

    def reset(self):
        """Khôi phục trạng thái ban đầu của session: xóa sạch ring buffer và lịch sử xác suất."""
        self.ring_buffer = AudioRingBuffer(sample_rate=self.sample_rate)
        self.full_probs_so_far = None
        self.chunk_count = 0
        self.active_speaker_history = []
        self.created_at = time.time()

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
        self._batch_queue: Optional[asyncio.Queue] = None
        self._batch_task: Optional[asyncio.Task] = None
        self._max_batch_size = 32
        self._batch_timeout_sec = 0.005

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

        # Tìm kiếm checkpoint model SWA xuất xưởng
        candidates = [
            custom_model_path or "",
            str(ROOT_DIR / "checkpoints" / "model_swa_v1_3_SWA.pt"),
            str(SERVER_DIR / "checkpoints" / "model_swa_v1_3_SWA.pt"),
            "checkpoints/model_swa_v1_3_SWA.pt",
            "model_swa_v1_3_SWA.pt",
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
            payload = torch.load(resolved, map_location="cpu", weights_only=True)
            state_dict = payload.get("model_state_dict", payload.get("state_dict", payload))
            incompatible = self.model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                print(f"[!] Missing keys ({len(incompatible.missing_keys)}): {incompatible.missing_keys[:5]}")
            if incompatible.unexpected_keys:
                print(f"[!] Unexpected keys ({len(incompatible.unexpected_keys)}): {incompatible.unexpected_keys[:5]}")
            self.model_path = resolved
        else:
            print("[!] Cảnh báo: Sử dụng Pretrained base weights (không tìm thấy checkpoint v1.3 local).")

        self.model.to(self.device)
        self.model.eval()

        if self.device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("[*] CUDA deterministic mode: ON (đảm bảo kết quả suy luận nhất quán giữa các lần chạy)")

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

    # --- Dynamic Batching ---

    async def start_batch_worker(self):
        self._batch_queue = asyncio.Queue()
        self._batch_task = asyncio.create_task(self._batch_loop())
        print("[*] Dynamic Batch Worker đã khởi động.")

    async def stop_batch_worker(self):
        if self._batch_task:
            self._batch_task.cancel()
            try:
                await self._batch_task
            except asyncio.CancelledError:
                pass
        if self._batch_queue:
            while not self._batch_queue.empty():
                try:
                    _, fut = self._batch_queue.get_nowait()
                    fut.set_exception(RuntimeError("Server shutting down"))
                except asyncio.QueueEmpty:
                    break
        print("[*] Dynamic Batch Worker đã dừng.")

    async def submit_for_inference(self, audio_slice: np.ndarray) -> Tuple[np.ndarray, float]:
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self._batch_queue.put((audio_slice, future))
        return await future

    async def _batch_loop(self):
        while True:
            first_item = await self._batch_queue.get()
            items = [first_item]
            deadline = asyncio.get_event_loop().time() + self._batch_timeout_sec
            while len(items) < self._max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(
                        self._batch_queue.get(), timeout=remaining
                    )
                    items.append(item)
                except asyncio.TimeoutError:
                    break

            audio_slices = [item[0] for item in items]
            futures = [item[1] for item in items]
            try:
                results = await asyncio.to_thread(self._forward_batch, audio_slices)
                for fut, result in zip(futures, results):
                    if not fut.done():
                        fut.set_result(result)
            except Exception as exc:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)

    def _forward_batch(self, audio_slices: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        if len(audio_slices) == 1:
            return [self.forward_audio_slice(audio_slices[0])]

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        processed = [
            self.processor(s, sampling_rate=DEFAULT_SAMPLE_RATE, return_tensors="pt")
            for s in audio_slices
        ]
        max_len = max(p["input_values"].shape[-1] for p in processed)

        padded_vals = []
        mask_vals = []
        for p in processed:
            v = p["input_values"][0]
            cur_len = v.shape[-1]
            if cur_len < max_len:
                pad = torch.zeros(max_len - cur_len, dtype=v.dtype)
                v = torch.cat([v, pad])
                m = torch.cat([torch.ones(cur_len), torch.zeros(max_len - cur_len)])
            else:
                m = torch.ones(cur_len)
            padded_vals.append(v)
            mask_vals.append(m)

        batch_input = torch.stack(padded_vals).to(self.device)
        batch_mask = torch.stack(mask_vals).to(self.device)

        with torch.inference_mode():
            logits = self.model(input_values=batch_input, attention_mask=batch_mask).logits
            probs = torch.sigmoid(logits).detach().cpu().numpy()

        if self.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        per_item_latency = latency_ms / len(audio_slices)
        return [(probs[i], per_item_latency) for i in range(len(audio_slices))]


# ===============================================================================
# 5. KHỞI TẠO ỨNG DỤNG FASTAPI & SESSION MANAGER
# ===============================================================================

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    manager = ModelManager.get_instance()
    manager.load_model()
    await manager.start_batch_worker()
    yield
    await manager.stop_batch_worker()

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

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

active_sessions: Dict[str, DiarizationSession] = {}


def parse_rttm_text(rttm_text: str, uri: str = "eval_session"):
    """Đọc chuỗi RTTM chuẩn NIST thành pyannote.core.Annotation."""
    from pyannote.core import Annotation, Segment
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
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
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
    threshold: float = Query(DEFAULT_THRESHOLD),
    key: Optional[str] = Query(None)
):
    """
    Kênh WebSocket truyền nhận âm thanh 2 chiều:
    - Client gửi:
        + Binary frame: Mảng bytes PCM 16-bit 16kHz Mono (mỗi gói khoảng 10ms - 240ms).
        + Text frame JSON: {"action": "reset"} hoặc {"action": "ping"}.
    - Server trả về JSON Event ngay khi có chunk 240ms hoàn thành.
    """
    await websocket.accept()

    if DIARIZATION_API_KEY and key != DIARIZATION_API_KEY:
        await websocket.send_json({"type": "error", "message": "Invalid or missing API key"})
        await websocket.close(code=4001)
        return

    if len(active_sessions) >= MAX_CONCURRENT_SESSIONS:
        await websocket.send_json({"type": "error", "message": f"Server at maximum capacity ({MAX_CONCURRENT_SESSIONS} sessions)"})
        await websocket.close(code=4002)
        return

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
                if len(raw_bytes) > MAX_WS_FRAME_BYTES:
                    await websocket.send_json({"type": "error", "message": "Frame size exceeds 1MB limit"})
                    continue
                session.ring_buffer.append_pcm16_bytes(raw_bytes)

                # 2. Kiểm tra xem đã đủ mẫu để chạy 1 chunk 240ms chưa
                while session.ring_buffer.has_enough_for_chunk(session.chunk_samples, session.lookahead_samples):
                    audio_slice, offset_samples, emitted_len = session.get_inference_window()

                    raw_probs, infer_latency = await manager.submit_for_inference(audio_slice)

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
                        session.reset()
                        await websocket.send_json({"type": "reset_ack", "session_id": session_id})
                        print(f"[*] Phiên {session_id} đã RESET sạch trạng thái về 0.0s")
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

@app.get("/", response_class=FileResponse)
def get_demo_ui():
    """Phục vụ giao diện web trực quan tách biệt từ scripts/04_deployment/frontend/index.html."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>404: Không tìm thấy frontend/index.html</h1>", status_code=404)


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
