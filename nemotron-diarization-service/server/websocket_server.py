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

try:
    import soxr
except ImportError:  # resample phía server sẽ bị vô hiệu hoá
    soxr = None

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
DEFAULT_CHUNK_DURATION = 0.24       # 240ms bước nhảy (3 encoder frames)
DEFAULT_LOOKAHEAD_DURATION = 0.08   # 80ms nhìn trước (1 encoder frame, tổng SLA = 320ms)
DEFAULT_STREAMING_MODE = "ultra_low_latency"  # (chunk 3, right context 1) — đúng cấu hình fine-tune
DEFAULT_THRESHOLD = 0.65            # Ngưỡng kích hoạt tiếng nói tối ưu
DEFAULT_MIN_SPEECH_ON = 0.20        # Lọc nhiễu < 50ms
DEFAULT_MIN_GAP_OFF = 0.15          # Nối khoảng ngắt < 150ms

DIARIZATION_API_KEY = os.environ.get("DIARIZATION_API_KEY", "")
MAX_CONCURRENT_SESSIONS = int(os.environ.get("MAX_CONCURRENT_SESSIONS", "50"))
MAX_WS_FRAME_BYTES = 1024 * 1024   # 1MB max per WebSocket binary frame


# ===============================================================================
# 2. HẬU XỬ LÝ TÍN HIỆU (ĐỒNG BỘ 1:1 VỚI PIPELINE ĐÁNH GIÁ LÚC TRAIN)
# ===============================================================================

def binarize_and_filter_probs(
    probs: np.ndarray,
    frame_step: float = 0.01,
    threshold: float = DEFAULT_THRESHOLD,
    min_duration_on: float = DEFAULT_MIN_SPEECH_ON,
    min_duration_off: float = DEFAULT_MIN_GAP_OFF
) -> List[Tuple[float, float, str]]:
    """
    Chuyển ma trận xác suất (T, 8) thành danh sách đoạn thoại (start, end, speaker_id).

    Bản sao chính xác của `scripts/02_baseline_eval/tune_postprocessing.py`
    để kết quả của service trùng khớp từng frame với pipeline đánh giá lúc train:
      1. Binarize ở cấp frame 10ms: probs >= threshold
      2. Gap bridging: nối các khoảng lặng ngắn hơn min_duration_off
      3. Short speech filtering: bỏ các đoạn ngắn hơn min_duration_on

    Lưu ý về tính "streaming": bước 2 chỉ nhìn lại tối đa `min_duration_off` (150ms)
    và bước 3 tối đa `min_duration_on` (50ms), nên bộ lọc này hoàn toàn chạy được
    online với độ trễ đầu ra 150ms. Nó KHÔNG dùng thông tin tương lai xa.
    """
    num_frames, max_speakers = probs.shape
    segments: List[Tuple[float, float, str]] = []

    for spk_idx in range(max_speakers):
        active = (probs[:, spk_idx] >= threshold)
        if not np.any(active):
            continue

        diff = np.diff(active.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if active[0]:
            starts = np.r_[0, starts]
        if active[-1]:
            ends = np.r_[ends, num_frames]

        spk_segments = []
        for s, e in zip(starts, ends):
            start_sec, end_sec = s * frame_step, e * frame_step
            if end_sec > start_sec:
                spk_segments.append([start_sec, end_sec])
        if not spk_segments:
            continue

        merged = [spk_segments[0]]
        for cur in spk_segments[1:]:
            if cur[0] - merged[-1][1] < min_duration_off:
                merged[-1][1] = cur[1]
            else:
                merged.append(cur)

        for seg in merged:
            if seg[1] - seg[0] >= min_duration_on:
                segments.append((seg[0], seg[1], f"SPEAKER_{spk_idx:02d}"))

    return segments


def segments_to_rttm_text(segments: List[Tuple[float, float, str]], uri: str) -> str:
    """Ghi danh sách segment ra chuỗi RTTM chuẩn NIST 10 cột."""
    lines = []
    for start, end, spk in sorted(segments, key=lambda x: x[0]):
        duration = max(0.01, end - start)
        lines.append(f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {spk} <NA> <NA>")
    return "\n".join(lines) + ("\n" if lines else "")


# ===============================================================================
# 3. STREAMING DIARIZATION SESSION (NATIVE SPEAKER CACHE — KHÔNG HUNGARIAN NGOÀI)
# ===============================================================================

class DiarizationSession:
    """
    Trạng thái phân đoạn người nói của 1 phiên, dùng ĐÚNG streaming API gốc của
    Nemotron-3-Diarization ở chế độ `ultra_low_latency` (chunk 3 frames = 240ms,
    right context 1 frame = 80ms → SLA 320ms):

      - Mỗi bước, processor được gọi với `is_streaming=True` và model nhận lại
        `speaker_cache` của bước trước. Speaker cache (264 frames) + FIFO (264 frames)
        do chính model duy trì, nên danh tính 8 kênh được giữ nguyên xuyên suốt phiên.
      - KHÔNG còn cửa sổ trượt 2.64s, KHÔNG còn Hungarian matching bên ngoài:
        cả hai chỉ là cách vá lỗi cho việc speaker cache bị tạo mới ở mỗi chunk.
      - Giữ lại toàn bộ xác suất frame 10ms của phiên để xuất RTTM cuối phiên
        bằng đúng bộ lọc hậu xử lý của pipeline đánh giá lúc train.
    """

    def __init__(
        self,
        session_id: str,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        threshold: float = DEFAULT_THRESHOLD,
        min_duration_on: float = DEFAULT_MIN_SPEECH_ON,
        min_duration_off: float = DEFAULT_MIN_GAP_OFF,
        emit_frame_probs: bool = False,
        input_sample_rate: int = DEFAULT_SAMPLE_RATE
    ):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.input_sample_rate = int(input_sample_rate)
        self.threshold = threshold
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.emit_frame_probs = emit_frame_probs

        manager = ModelManager.get_instance()
        # Kích thước chunk do chính processor quy định cho chế độ ultra_low_latency
        self.samples_first_chunk = manager.samples_first_chunk   # 5,160 samples
        self.samples_per_chunk = manager.samples_per_chunk       # 5,520 samples
        self.samples_per_step = manager.samples_per_step         # 3,840 samples (240ms)
        self.frames_per_step = manager.frames_per_step           # 24 frames (10ms/frame)

        self.reset()

    def reset(self):
        """Khôi phục trạng thái ban đầu: xóa sạch audio đệm, speaker cache và lịch sử."""
        self.pending = np.zeros(0, dtype=np.float32)
        self.speaker_cache = None
        self.is_first_chunk = True
        self.finished = False
        self.frames_emitted = 0
        self.session_probs: List[np.ndarray] = []
        self.chunk_count = 0
        self.created_at = time.time()
        self.active_speaker_history: List[str] = []
        self._build_resampler()

    def _build_resampler(self):
        """
        Dựng bộ resample streaming (soxr, chất lượng HQ) nếu client gửi audio
        ở sample rate khác 16kHz.

        Resample PHẢI làm ở server: trình duyệt dùng nội suy tuyến tính
        (AudioBufferSourceNode trong OfflineAudioContext), gây méo phổ và làm
        DER xấu đi rõ rệt trên audio điện thoại 8kHz. Dùng soxr ở đây đảm bảo
        web UI và client Python đi qua đúng một đường xử lý tín hiệu.
        """
        self._resampler = None
        if self.input_sample_rate == self.sample_rate:
            return
        if soxr is None:
            raise RuntimeError(
                f"Client gửi audio {self.input_sample_rate}Hz nhưng server thiếu gói `soxr`. "
                f"Cài `pip install soxr` hoặc gửi audio đã resample sẵn về {self.sample_rate}Hz."
            )
        self._resampler = soxr.ResampleStream(
            self.input_sample_rate, self.sample_rate, 1, dtype="float32", quality="HQ"
        )

    def set_input_sample_rate(self, rate: int):
        """Đổi sample rate đầu vào (chỉ hợp lệ trước khi nhận audio của phiên)."""
        rate = int(rate)
        if rate <= 0:
            raise ValueError(f"sample_rate không hợp lệ: {rate}")
        if self.frames_emitted > 0 or len(self.pending) > 0:
            raise RuntimeError("Phải đặt sample_rate TRƯỚC khi gửi audio (gửi 'reset' để bắt đầu lại).")
        self.input_sample_rate = rate
        self._build_resampler()

    # --- Nạp audio -------------------------------------------------------------

    def append_pcm16_bytes(self, pcm_bytes: bytes) -> int:
        """Nhận PCM 16-bit little endian mono và chuyển sang float32 [-1.0, 1.0]."""
        if not pcm_bytes:
            return 0
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        return self.append_float32(audio_int16.astype(np.float32) / 32768.0)

    def append_float32(self, samples: np.ndarray, last: bool = False) -> int:
        if len(samples) == 0 and not last:
            return 0
        x = samples.astype(np.float32)
        if self._resampler is not None:
            x = self._resampler.resample_chunk(x, last=last)
        if len(x) == 0:
            return 0
        self.pending = np.concatenate([self.pending, x])
        return len(x)

    def flush_resampler(self):
        """Đẩy nốt phần đuôi còn kẹt trong bộ lọc resample (gọi khi finalize)."""
        if self._resampler is not None:
            self.append_float32(np.zeros(0, dtype=np.float32), last=True)

    # --- Điều phối chunk -------------------------------------------------------

    def _samples_needed(self) -> int:
        return self.samples_first_chunk if self.is_first_chunk else self.samples_per_chunk

    def has_enough_for_chunk(self) -> bool:
        """Đã đủ audio cho 1 bước streaming (chunk 240ms + lookahead 80ms) chưa."""
        return (not self.finished) and len(self.pending) >= self._samples_needed()

    def next_chunk_audio(self, is_last: bool = False) -> np.ndarray:
        """Lấy đoạn audio cho bước suy luận kế tiếp (chưa tiêu thụ)."""
        if is_last:
            return self.pending
        return self.pending[: self._samples_needed()]

    def consume_step(self):
        """Tiêu thụ đúng 240ms audio sau khi một bước streaming hoàn tất."""
        self.pending = self.pending[self.samples_per_step:]
        self.is_first_chunk = False

    # --- Ghi nhận kết quả 1 bước ----------------------------------------------

    def record_chunk(self, frame_probs: np.ndarray) -> Tuple[float, float, List[Dict[str, Any]]]:
        """
        Ghi nhận ma trận xác suất frame 10ms của 1 bước và trả về
        (t_start, t_end, active_speakers) cho sự kiện real-time.
        """
        n_frames = len(frame_probs)
        t_start = self.frames_emitted * 0.01
        t_end = (self.frames_emitted + n_frames) * 0.01

        self.session_probs.append(frame_probs)
        self.frames_emitted += n_frames
        self.chunk_count += 1

        # Quyết định ở cấp frame rồi mới tổng hợp cho chunk: một tiếng động 40ms
        # không còn kéo cả khối 240ms thành "đang nói" như cách lấy mean trước đây.
        active_mask = frame_probs >= self.threshold
        active_speakers: List[Dict[str, Any]] = []
        for spk_idx in range(frame_probs.shape[1]):
            active_ratio = float(np.mean(active_mask[:, spk_idx])) if n_frames else 0.0
            if active_ratio >= 0.5:
                spk_name = f"SPEAKER_{spk_idx:02d}"
                active_speakers.append({
                    "speaker_id": spk_name,
                    "channel_index": spk_idx,
                    "confidence": round(float(np.mean(frame_probs[:, spk_idx])), 4),
                    "active_frame_ratio": round(active_ratio, 3)
                })
                if spk_name not in self.active_speaker_history:
                    self.active_speaker_history.append(spk_name)

        return t_start, t_end, active_speakers

    # --- Kết thúc phiên --------------------------------------------------------

    def build_final_rttm(self, uri: Optional[str] = None) -> Tuple[str, List[Tuple[float, float, str]]]:
        """
        Xuất RTTM cuối phiên từ toàn bộ xác suất frame 10ms, dùng đúng bộ lọc
        (threshold / min_duration_on / min_duration_off) của pipeline lúc train.
        """
        if not self.session_probs:
            return "", []
        probs = np.concatenate(self.session_probs, axis=0)
        segments = binarize_and_filter_probs(
            probs,
            frame_step=0.01,
            threshold=self.threshold,
            min_duration_on=self.min_duration_on,
            min_duration_off=self.min_duration_off,
        )
        return segments_to_rttm_text(segments, uri or self.session_id), segments


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
        # Kích thước chunk của chế độ ultra_low_latency, điền khi load_model()
        self.samples_first_chunk = 0
        self.samples_per_chunk = 0
        self.samples_per_step = 0
        self.frames_per_step = 0
        self._infer_lock: Optional[asyncio.Lock] = None

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

        # BẮT BUỘC: chọn đúng chế độ streaming 0.32s mà model đã được fine-tune.
        # Mặc định của processor là "low_latency" (9 frames chunk / 4 frames lookahead
        # = 1040ms), KHÔNG phải cấu hình 0.32s dùng khi train.
        self.processor.set_streaming_mode("ultra_low_latency")
        self.samples_first_chunk = self.processor.num_samples_first_audio_chunk
        self.samples_per_chunk = self.processor.num_samples_per_audio_chunk
        self.frames_per_step = self.processor.num_mel_frames_per_step
        self.samples_per_step = self.frames_per_step * self.processor.feature_extractor.hop_length
        sc = self.model.config.streaming_config
        print(f"[*] Streaming mode: ultra_low_latency | chunk={self.processor._streaming_chunk_sizes[0]} frames "
              f"+ lookahead={self.processor._streaming_chunk_sizes[1]} frame "
              f"= {self.processor.streaming_latency_ms}ms SLA")
        print(f"[*] Speaker cache={sc.speaker_cache_length} frames | FIFO={sc.fifo_length} frames | "
              f"update_period={sc.speaker_cache_update_period} | num_speakers={sc.num_speakers}")

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

        # Warm-up GPU bằng đúng đường đi streaming thật (chunk đầu + 2 chunk tiếp)
        print(f"[*] Thiết bị suy luận: {self.device.upper()}")
        self.model_loaded = True
        probs, cache, _ = self.forward_streaming_chunk(
            np.zeros(self.samples_first_chunk, dtype=np.float32), None, is_first=True, is_last=False)
        for _ in range(2):
            probs, cache, _ = self.forward_streaming_chunk(
                np.zeros(self.samples_per_chunk, dtype=np.float32), cache, is_first=False, is_last=False)
        if self.device == "cuda":
            torch.cuda.synchronize()
        print(f"[*] Warm-up OK: mỗi bước trả về {len(probs)} frames (10ms/frame = 240ms audio)")

        print("[+] MODEL STREAMING ĐÃ SẴN SÀNG PHỤC VỤ (WARM-UP HOÀN TẤT)!\n")

    def forward_streaming_chunk(
        self,
        audio_chunk: np.ndarray,
        speaker_cache: Any,
        is_first: bool,
        is_last: bool
    ) -> Tuple[np.ndarray, Any, float]:
        """
        Một bước streaming chuẩn Nemotron-3: nạp chunk audio kèm `speaker_cache`
        của bước trước, trả về (frame_probs (n,8), speaker_cache mới, latency_ms).

        `speaker_cache` chính là thứ giữ danh tính người nói ổn định xuyên suốt
        phiên — thay thế hoàn toàn Hungarian matching thủ công trước đây.
        """
        if not self.model_loaded:
            raise RuntimeError("Model chưa được nạp!")

        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            inputs = self.processor(
                audio_chunk,
                sampling_rate=DEFAULT_SAMPLE_RATE,
                return_tensors="pt",
                is_streaming=True,
                is_first_audio_chunk=is_first,
                is_last_audio_chunk=is_last,
            ).to(self.device)

            extra = {}
            if "num_lookahead_frames" in inputs:
                extra["num_lookahead_frames"] = int(inputs.pop("num_lookahead_frames"))

            out = self.model(**inputs, speaker_cache=speaker_cache, **extra)
            probs = torch.sigmoid(out.logits)[0].float().detach().cpu().numpy()
            new_cache = out.speaker_cache

        if self.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        return probs, new_cache, (t1 - t0) * 1000.0

    # --- Điều phối suy luận ---
    #
    # Dynamic batching (gộp nhiều phiên vào 1 forward) đã được gỡ bỏ: ở chế độ
    # streaming thật, mỗi phiên mang một `speaker_cache` riêng với nội dung khác
    # nhau, nên không thể gộp các phiên vào cùng một batch mà không trộn lẫn
    # trạng thái người nói giữa các cuộc gọi. Thay vào đó, các forward được
    # tuần tự hoá bằng lock và chạy trong thread pool để không chẹn event loop.

    async def submit_streaming_chunk(
        self,
        audio_chunk: np.ndarray,
        speaker_cache: Any,
        is_first: bool,
        is_last: bool
    ) -> Tuple[np.ndarray, Any, float]:
        if self._infer_lock is None:
            self._infer_lock = asyncio.Lock()
        async with self._infer_lock:
            return await asyncio.to_thread(
                self.forward_streaming_chunk, audio_chunk, speaker_cache, is_first, is_last
            )


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
    description="Kiến trúc 1: WebSocket + Native Speaker Cache cho Realtime Diarization (SLA 0.32s)",
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
    min_duration_on: float = Query(DEFAULT_MIN_SPEECH_ON),
    min_duration_off: float = Query(DEFAULT_MIN_GAP_OFF),
    emit_frame_probs: bool = Query(False),
    input_sample_rate: int = Query(DEFAULT_SAMPLE_RATE),
    key: Optional[str] = Query(None)
):
    """
    Kênh WebSocket truyền nhận âm thanh 2 chiều:
    - Client gửi:
        + Binary frame: Mảng bytes PCM 16-bit 16kHz Mono (mỗi gói khoảng 10ms - 240ms).
        + Text frame JSON: {"action": "reset"} | {"action": "ping"} | {"action": "finalize"}.
    - Server trả về JSON Event ngay khi có chunk 240ms hoàn thành.
    - {"action": "finalize"} xử nốt phần audio dư rồi trả về `final_rttm` — RTTM của
      cả phiên, dựng từ xác suất frame 10ms qua đúng bộ lọc hậu xử lý lúc train.
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

    manager = ModelManager.get_instance()
    session = DiarizationSession(
        session_id=session_id,
        threshold=threshold,
        min_duration_on=min_duration_on,
        min_duration_off=min_duration_off,
        emit_frame_probs=emit_frame_probs,
        input_sample_rate=input_sample_rate,
    )
    active_sessions[session_id] = session

    async def run_step(is_last: bool = False) -> bool:
        """Chạy 1 bước streaming và gửi sự kiện về client. Trả về False nếu không chạy được."""
        audio_chunk = session.next_chunk_audio(is_last=is_last)
        if is_last and len(audio_chunk) < 400:   # ngắn hơn 1 cửa sổ phân tích mel
            return False

        frame_probs, new_cache, infer_latency = await manager.submit_streaming_chunk(
            audio_chunk, session.speaker_cache, session.is_first_chunk, is_last
        )
        session.speaker_cache = new_cache
        t_start, t_end, active_speakers = session.record_chunk(frame_probs)
        if is_last:
            session.finished = True
        else:
            session.consume_step()

        payload = {
            "type": "diarization_event",
            "session_id": session_id,
            "chunk_index": session.chunk_count,
            "time_start_sec": round(t_start, 3),
            "time_end_sec": round(t_end, 3),
            "active_speakers": active_speakers,
            "num_active_speakers": len(active_speakers),
            "unique_speakers_seen": list(session.active_speaker_history),
            "latency_ms": round(infer_latency, 2),
            "sla_passed": infer_latency <= 320.0,
            "is_final_chunk": is_last,
        }
        if session.emit_frame_probs:
            payload["frame_probs"] = np.round(frame_probs, 4).tolist()
        await websocket.send_json(payload)
        return True

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
                session.append_pcm16_bytes(raw_bytes)

                # 2. Chạy hết các bước streaming 240ms đã đủ dữ liệu
                while session.has_enough_for_chunk():
                    await run_step(is_last=False)

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
                    elif action == "config":
                        # Khai báo sample rate gốc của audio; server sẽ tự resample
                        # bằng soxr HQ — đúng một đường xử lý với client Python.
                        session.set_input_sample_rate(data.get("sample_rate", DEFAULT_SAMPLE_RATE))
                        await websocket.send_json({
                            "type": "config_ack",
                            "session_id": session_id,
                            "input_sample_rate": session.input_sample_rate,
                            "server_resampling": session.input_sample_rate != session.sample_rate,
                        })
                        print(f"[*] Phiên {session_id}: nhận audio {session.input_sample_rate}Hz"
                              + (" (server resample -> 16kHz, soxr HQ)"
                                 if session.input_sample_rate != session.sample_rate else ""))
                    elif action == "finalize":
                        # Đẩy nốt đuôi bộ lọc resample rồi xử nốt audio dư
                        session.flush_resampler()
                        while session.has_enough_for_chunk():
                            await run_step(is_last=False)
                        if not session.finished:
                            await run_step(is_last=True)
                        uri = data.get("uri") or session_id
                        rttm_text, segments = session.build_final_rttm(uri=uri)
                        await websocket.send_json({
                            "type": "final_rttm",
                            "session_id": session_id,
                            "uri": uri,
                            "rttm": rttm_text,
                            "num_segments": len(segments),
                            "total_frames": session.frames_emitted,
                            "duration_sec": round(session.frames_emitted * 0.01, 3),
                            "speakers": sorted({s[2] for s in segments}),
                            "post_processing": {
                                "threshold": session.threshold,
                                "min_duration_on": session.min_duration_on,
                                "min_duration_off": session.min_duration_off,
                            },
                        })
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
    """
    Phục vụ giao diện web, gắn thêm dấu vân tay (mtime) vào CSS/JS.

    Không có bước này, trình duyệt có thể giữ bản app.js cũ trong bộ nhớ đệm
    sau khi server đã cập nhật, khiến UI chạy logic cũ và cho ra metrics khác
    hẳn client Python — rất khó phát hiện vì mọi thứ trông vẫn bình thường.
    """
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h1>404: Không tìm thấy frontend/index.html</h1>", status_code=404)

    html = index_file.read_text(encoding="utf-8")
    for rel in ("css/style.css", "js/app.js"):
        asset = FRONTEND_DIR / rel
        if asset.exists():
            html = html.replace(f"/static/{rel}", f"/static/{rel}?v={int(asset.stat().st_mtime)}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache, must-revalidate"})


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
