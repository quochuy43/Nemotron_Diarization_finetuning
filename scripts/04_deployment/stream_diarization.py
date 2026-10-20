#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script: stream_diarization.py (Ứng dụng Streaming Diarization Thời Gian Thực)
Mục đích:
  - Cho phép người dùng chạy suy luận trực tiếp trên 1 file âm thanh bất kỳ
    (hoặc luồng âm thanh live) theo chuẩn Streaming Ultra-Low Latency 0.32s.
  - In ra màn hình console theo thời gian thực: Ai đang nói lúc nào.
  - Xuất file kết quả RTTM hoặc JSON.
===============================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModelForAudioFrameClassification

current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir.parent
for p in [str(scripts_dir), str(scripts_dir / "02_baseline_eval"), str(scripts_dir / "03_training")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate_metrics import resolve_file_path
from tune_postprocessing import binarize_and_filter_probs, write_segments_to_rttm
from evaluate_streaming import StreamingDiarizationEngine, find_checkpoint


def run_live_stream_inference(
    audio_path: str,
    model_path: Optional[str] = None,
    output_rttm: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    simulate_realtime: bool = True,
    mode: str = "streaming",
    threshold: float = 0.45,
    min_speech_on: float = 0.05,
    min_gap_off: float = 0.15
):
    print("=" * 80)
    print("🎙️ HỆ THỐNG PHÂN TÁCH NGƯỜI NÓI THỜI GIAN THỰC (NEMOTRON-3 STREAMING 0.32S)")
    print("=" * 80)

    # 1. Dò tìm model
    candidates = [
        model_path or "",
        "/content/drive/MyDrive/VJ_Technologies/Nemotron_Diarization/experiments/checkpoints/train_v1_3_ratio_1_8_swa_model/model_swa_v1_3_SWA.pt",
        "/content/drive/MyDrive/VJ_Technologies/Nemotron_Diarization/experiments/checkpoints/train_v1_3_ratio_1_8_swa_model/model_swa.pt",
        "experiments/checkpoints/train_v1_3_ratio_1_8_swa_model/model_swa_v1_3_SWA.pt",
        "experiments/checkpoints/train_v1_3_ratio_1_8_swa_model/model_swa.pt",
        "/content/drive/MyDrive/VJ_Technologies/Nemotron_Diarization/experiments/checkpoints/v1_3_swa_model/model_swa.pt"
    ]
    resolved_model = find_checkpoint(candidates)
    if not resolved_model:
        raise FileNotFoundError(f"Không tìm thấy mô hình! Vui lòng truyền đường dẫn bằng --model.")

    print(f"[*] Đang sử dụng mô hình: {resolved_model}")
    print(f"[*] File âm thanh đầu vào: {audio_path}")
    print(f"[*] Thiết bị (Device):     {device}")
    print(f"[*] Chế độ suy luận:       {mode.upper()}")
    print(f"[*] Siêu tham số:          Threshold={threshold}, MinOn={min_speech_on}s, MinOff={min_gap_off}s")

    pretrained_id = "nvidia/Nemotron-3-Diarization"
    processor = AutoProcessor.from_pretrained(pretrained_id)
    model = AutoModelForAudioFrameClassification.from_pretrained(pretrained_id)

    payload = torch.load(resolved_model, map_location="cpu")
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 2. Xử lý âm thanh
    resolved_audio = resolve_file_path(audio_path)
    engine = StreamingDiarizationEngine(model=model, processor=processor, device=device)

    print(f"\n[*] Bắt đầu xử lý âm thanh ({mode.upper()})...")
    print("-" * 80)
    t_start = time.time()
    if mode == "offline":
        probs, infer_ms = engine.process_file_offline(resolved_audio)
        latencies = [infer_ms]
    else:
        probs, latencies = engine.process_file_streaming(resolved_audio)
    total_time = time.time() - t_start

    info = sf.info(resolved_audio)
    dur = info.duration
    rtf = total_time / max(1e-5, dur)

    # 3. Hậu xử lý thành các phân đoạn người nói
    segs = binarize_and_filter_probs(
        probs,
        frame_step=0.01,
        threshold=threshold,
        min_duration_on=min_speech_on,
        min_duration_off=min_gap_off
    )

    print("\n📋 KẾT QUẢ PHÂN PHỐI LƯỢT NÓI (DIARIZATION TIMESTAMPS):")
    print("-" * 80)
    for seg in segs:
        t_on = seg["start"]
        t_off = seg["end"]
        spk = seg["speaker_id"]
        print(f"  ⏱️  [{t_on:06.2f}s  -->  {t_off:06.2f}s]  |  🔊 Người nói: {spk}  (Thời lượng: {t_off - t_on:.2f}s)")

    print("-" * 80)
    print(f"[*] Tổng thời lượng âm thanh: {dur:.2f}s")
    print(f"[*] Thời gian xử lý:         {total_time:.2f}s")
    print(f"[*] Real-Time Factor (RTF):   {rtf:.4f} ({1.0/max(1e-5, rtf):.1f}x tốc độ thực tế)")
    if latencies and mode == "streaming":
        print(f"[*] Độ trễ mỗi chunk 320ms:   p50={np.percentile(latencies, 50):.1f}ms | p99={np.percentile(latencies, 99):.1f}ms")

    if output_rttm:
        out_p = Path(output_rttm)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        write_segments_to_rttm(segs, uri=Path(audio_path).stem, rttm_path=str(out_p))
        print(f"[+] Đã lưu file kết quả RTTM tại: {out_p}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Streaming Diarization thời gian thực trên 1 file âm thanh")
    parser.add_argument("--audio", type=str, required=True, help="Đường dẫn file WAV âm thanh 16kHz")
    parser.add_argument("--model", type=str, default=None, help="Đường dẫn file checkpoint model SWA")
    parser.add_argument("--mode", type=str, choices=["streaming", "offline"], default="streaming", help="Chế độ suy luận ('streaming' có FIFO context & khóa kênh hoặc 'offline')")
    parser.add_argument("--output_rttm", type=str, default=None, help="Đường dẫn lưu file RTTM đầu ra")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị (cuda/cpu)")
    parser.add_argument("--threshold", type=float, default=0.45, help="Ngưỡng quyết định")
    parser.add_argument("--min_on", type=float, default=0.05, help="Thời lượng nói tối thiểu (s)")
    parser.add_argument("--min_off", type=float, default=0.15, help="Khoảng lặng ngắt nghỉ (s)")

    args = parser.parse_args()

    run_live_stream_inference(
        audio_path=args.audio,
        model_path=args.model,
        output_rttm=args.output_rttm,
        device=args.device,
        mode=args.mode,
        threshold=args.threshold,
        min_speech_on=args.min_on,
        min_gap_off=args.min_off
    )
