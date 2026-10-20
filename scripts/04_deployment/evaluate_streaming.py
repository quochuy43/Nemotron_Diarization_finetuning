#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script: evaluate_streaming.py (Giai đoạn Triển khai & Kiểm chứng Streaming)
Mục đích:
  1. Mô phỏng luồng suy luận Streaming Ultra-Low Latency 0.32s (320ms buffer)
     chunk-by-chunk chuẩn xác theo kiến trúc NVIDIA Nemotron-3 Diarization.
  2. Đo đạc phân phối độ trễ thời gian thực ở cấp độ từng chunk (p50, p90, p95, p99, max, RTF)
     để kiểm chứng cam kết chất lượng dịch vụ (SLA: p99 <= 320ms).
  3. Chạy đánh giá đối đầu trực tiếp giữa 2 phiên bản xuất sắc nhất (v1.1 vs v1.3)
     trên toàn bộ 43 files của Frozen Hold-out Test Set (Zero Data Leakage).
  4. Xuất bảng so sánh tổng hợp chi tiết và file báo cáo JSON.
===============================================================================
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForAudioFrameClassification

# Đưa thư mục cha và 02_baseline_eval vào sys.path
current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir.parent
baseline_eval_dir = scripts_dir / "02_baseline_eval"
training_dir = scripts_dir / "03_training"

for p in [str(scripts_dir), str(baseline_eval_dir), str(training_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate_metrics import compute_metrics_for_dataset, resolve_file_path
from tune_postprocessing import binarize_and_filter_probs, write_segments_to_rttm


# ===============================================================================
# 1. HÀM TỰ ĐỘNG DÒ TÌM CHECKPOINTS
# ===============================================================================

def find_checkpoint(candidates: List[str]) -> str:
    """Tự động tìm kiếm file checkpoint tồn tại đầu tiên trong danh sách ứng viên."""
    for cand in candidates:
        if not cand:
            continue
        p = Path(resolve_file_path(cand))
        if p.exists() and p.is_file():
            return str(p)
    return ""


# ===============================================================================
# 2. BỘ MÔ PHỎNG STREAMING INFERENCE 0.32S (CHUNKING ENGINE)
# ===============================================================================

class StreamingDiarizationEngine:
    """
    Bộ suy luận Streaming thời gian thực:
      - Nhận audio theo từng chunk buffer = 0.32s (240ms chunk + 80ms lookahead).
      - Bước nhảy thời gian (hop) = 0.24s (240ms).
      - Đo đạc chính xác thời gian tính toán từng chunk (latency_ms).
    """
    def __init__(
        self,
        model: Any,
        processor: Any,
        device: str = "cuda",
        chunk_duration: float = 0.24,       # 240ms = 3 frames
        lookahead_duration: float = 0.08,   # 80ms = 1 frame lookahead
        sample_rate: int = 16000
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.lookahead_duration = lookahead_duration
        self.buffer_duration = chunk_duration + lookahead_duration  # 0.32s = 320ms

        self.chunk_samples = int(round(self.chunk_duration * self.sample_rate))        # 3,840 samples
        self.buffer_samples = int(round(self.buffer_duration * self.sample_rate))      # 5,120 samples
        self.lookahead_samples = self.buffer_samples - self.chunk_samples              # 1,280 samples

    def process_file_streaming(self, wav_path: str) -> Tuple[np.ndarray, List[float]]:
        """
        Đọc 1 file âm thanh và giả lập quá trình streaming chunk-by-chunk:
        Trả về:
          - full_probs: Ma trận xác suất người nói (T_frames, 8)
          - latencies_ms: Danh sách độ trễ tính toán của từng chunk (ms)
        """
        wav_data, sr = sf.read(wav_path)
        if wav_data.ndim > 1:
            wav_data = wav_data[:, 0]

        total_samples = len(wav_data)
        latencies_ms = []
        all_chunk_probs = []

        cur_start = 0
        with torch.inference_mode():
            while cur_start < total_samples:
                cur_end = min(total_samples, cur_start + self.buffer_samples)
                chunk_audio = wav_data[cur_start:cur_end]

                # Nếu chunk cuối ngắn hơn buffer, đệm zero cho đủ buffer_samples
                if len(chunk_audio) < self.buffer_samples:
                    pad_len = self.buffer_samples - len(chunk_audio)
                    chunk_audio = np.pad(chunk_audio, (0, pad_len), mode="constant")

                # Đo đạc thời gian suy luận (GPU Synchronize để đo chuẩn xác)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()

                inputs = self.processor(chunk_audio, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
                logits = self.model(**inputs).logits # Shape: (1, num_frames, 8)
                probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

                if self.device == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                latency_ms = (t1 - t0) * 1000.0
                latencies_ms.append(latency_ms)

                # Cắt bỏ phần dự đoán của lookahead để ghép nối liên tục
                # Tỷ lệ khung hình: 100 frames/sec (1 frame = 10ms)
                emitted_frames = int(round(self.chunk_duration * 100)) # 24 frames
                actual_chunk_len = min(cur_end - cur_start, self.chunk_samples)
                actual_frames = int(round((actual_chunk_len / self.sample_rate) * 100))

                valid_frames = probs[:min(actual_frames, emitted_frames)]
                if len(valid_frames) > 0:
                    all_chunk_probs.append(valid_frames)

                cur_start += self.chunk_samples

        if all_chunk_probs:
            full_probs = np.concatenate(all_chunk_probs, axis=0)
        else:
            full_probs = np.zeros((1, 8), dtype=np.float32)

        return full_probs, latencies_ms


# ===============================================================================
# 3. ĐÁNH GIÁ TOÀN DIỆN TRÊN TẬP DỮ LIỆU
# ===============================================================================

def run_streaming_benchmark_on_manifest(
    model: Any,
    processor: Any,
    manifest_path: str,
    output_rttm_dir: Path,
    device: str = "cuda",
    threshold: float = 0.45,
    min_duration_on: float = 0.05,
    min_duration_off: float = 0.15,
    limit_files: Optional[int] = None
) -> Dict[str, Any]:
    """
    Chạy đánh giá streaming trên manifest và tính toán toàn diện DER + Latency.
    """
    engine = StreamingDiarizationEngine(model=model, processor=processor, device=device)
    output_rttm_dir.mkdir(parents=True, exist_ok=True)

    manifest_p = Path(resolve_file_path(manifest_path))
    manifest_items = []
    with open(manifest_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_items.append(json.loads(line))

    if limit_files is not None and limit_files > 0:
        manifest_items = manifest_items[:limit_files]

    all_latencies_ms = []
    total_audio_duration_sec = 0.0
    total_inference_time_sec = 0.0

    print(f"[*] Đang chạy Streaming Benchmark trên {len(manifest_items)} files ({manifest_p.name})...")

    for item in tqdm(manifest_items, desc="Streaming Inference", unit="file"):
        wav_path = resolve_file_path(item["audio_filepath"])
        stem = Path(wav_path).stem
        duration = float(item.get("duration", 0.0))
        total_audio_duration_sec += duration

        t_file_start = time.perf_counter()
        probs, file_latencies = engine.process_file_streaming(wav_path)
        t_file_end = time.perf_counter()

        total_inference_time_sec += (t_file_end - t_file_start)
        all_latencies_ms.extend(file_latencies)

        # Hậu xử lý thành RTTM
        segs = binarize_and_filter_probs(
            probs,
            frame_step=0.01,
            threshold=threshold,
            min_duration_on=min_duration_on,
            min_duration_off=min_duration_off
        )
        out_rttm = output_rttm_dir / f"{stem}.rttm"
        write_segments_to_rttm(segs, uri=stem, rttm_path=str(out_rttm))

    # 1. Tính toán số liệu phân phối độ trễ (Latency Distribution)
    lat_arr = np.array(all_latencies_ms)
    p50 = float(np.percentile(lat_arr, 50))
    p90 = float(np.percentile(lat_arr, 90))
    p95 = float(np.percentile(lat_arr, 95))
    p99 = float(np.percentile(lat_arr, 99))
    mean_lat = float(np.mean(lat_arr))
    max_lat = float(np.max(lat_arr))
    rtf = total_inference_time_sec / max(1e-5, total_audio_duration_sec)

    latency_report = {
        "total_chunks_processed": len(lat_arr),
        "mean_chunk_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50, 2),
        "p90_latency_ms": round(p90, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "max_chunk_latency_ms": round(max_lat, 2),
        "real_time_factor_rtf": round(rtf, 4),
        "sla_p99_threshold_ms": 320.0,
        "sla_passed": p99 <= 320.0
    }

    # 2. Tính toán toàn diện chỉ số NIST (DER, JER, SCA, MAE)
    metrics = compute_metrics_for_dataset(
        manifest_path=str(manifest_p),
        hyp_rttm_dir=str(output_rttm_dir),
        collar_main=0.0,
        collar_secondary=0.25,
        verbose=False
    )

    return {
        "accuracy_metrics": metrics,
        "latency_metrics": latency_report
    }


# ===============================================================================
# 4. HÀM CHÍNH SO SÁNH ĐỐI ĐẦU V1.1 VS V1.3
# ===============================================================================

def run_streaming_comparison(
    model_v1_1_path: Optional[str] = None,
    model_v1_3_path: Optional[str] = None,
    test_manifest: str = "data/manifests/test_manifest.json",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "experiments/streaming_eval",
    limit_files: Optional[int] = None
):
    print("=" * 100)
    print("🚀 BẮT ĐẦU CHƯƠNG TRÌNH ĐÁNH GIÁ STREAMING INFERENCE 0.32S: V1.1 VS V1.3")
    print("=" * 100)
    print(f"[*] Thiết bị (Device):         {device}")
    print(f"[*] Tập Test Đóng Băng:       {test_manifest}")
    print(f"[*] Thư mục lưu kết quả:      {output_dir}")

    # 1. Dò tìm đường dẫn checkpoint của v1.1
    v1_1_candidates = [
        model_v1_1_path or "",
        "/content/drive/MyDrive/VJ_Technologies/Nemotron_Diarization/experiments/checkpoints/train_v1_1_ratio_1_3_swa_model/train_v1_1_ratio_1_3_swa_model.pt"
    ]
    resolved_v1_1 = find_checkpoint(v1_1_candidates)

    # 2. Dò tìm đường dẫn checkpoint của v1.3
    v1_3_candidates = [
        model_v1_3_path or "",
        "/content/drive/MyDrive/VJ_Technologies/Nemotron_Diarization/experiments/checkpoints/train_v1_3_ratio_1_8_swa_model/model_swa_v1_3_SWA.pt"
    ]
    resolved_v1_3 = find_checkpoint(v1_3_candidates)

    print(f"[*] Model v1.1 resolved:       {resolved_v1_1 or 'CHƯA TÌM THẤY'}")
    print(f"[*] Model v1.3 resolved:       {resolved_v1_3 or 'CHƯA TÌM THẤY'}")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    models_to_run = []
    if resolved_v1_1:
        models_to_run.append(("v1.1 (Tỷ lệ 1:3)", resolved_v1_1, out_p / "rttm_v1_1"))
    if resolved_v1_3:
        models_to_run.append(("v1.3 (Tỷ lệ 1:8)", resolved_v1_3, out_p / "rttm_v1_3"))

    if not models_to_run:
        print("\n❌ LỖI: Không tìm thấy checkpoint nào cho v1.1 hoặc v1.3!")
        print("Vui lòng truyền đường dẫn thủ công bằng --model_v1_1 hoặc --model_v1_3.")
        return

    results = {}
    pretrained_id = "nvidia/Nemotron-3-Diarization"
    processor = AutoProcessor.from_pretrained(pretrained_id)

    for tag, cp_path, rttm_sub_dir in models_to_run:
        print("\n" + "-" * 80)
        print(f"[*] Nạp mô hình {tag} từ: {cp_path}")
        model = AutoModelForAudioFrameClassification.from_pretrained(pretrained_id)

        payload = torch.load(cp_path, map_location="cpu")
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        else:
            state_dict = payload

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        res = run_streaming_benchmark_on_manifest(
            model=model,
            processor=processor,
            manifest_path=test_manifest,
            output_rttm_dir=rttm_sub_dir,
            device=device,
            limit_files=limit_files
        )
        results[tag] = res

        # Xóa model giải phóng VRAM GPU cho lượt kế tiếp
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # 3. In bảng so sánh đối đầu chi tiết
    print("\n" + "=" * 105)
    print("📊 BẢNG SO SÁNH ĐỐI ĐẦU CHÍNH THỨC: V1.1 VS V1.3 TRÊN STREAMING TEST SET (0.32S)")
    print("=" * 105)
    print(f"{'CHỈ SỐ ĐO ĐẠC':<32} | {'PHIÊN BẢN v1.1 (1:3)':<30} | {'PHIÊN BẢN v1.3 (1:8)':<30}")
    print("-" * 105)

    def get_val(res, path_keys, fmt="{:.2f}%"):
        cur = res
        for k in path_keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return "N/A"
        if isinstance(cur, (int, float)):
            return fmt.format(cur)
        return str(cur)

    metrics_rows = [
        ("DER Tổng Thể (Collar 0.0s)", ["accuracy_metrics", "overall", "der_collar0"], "{:.2f}%"),
        ("DER Tổng Thể (Collar 0.25s)", ["accuracy_metrics", "overall", "der_collar025"], "{:.2f}%"),
        ("  - Tỷ lệ Bỏ sót (Miss Rate c0)", ["accuracy_metrics", "overall", "miss_rate_c0"], "{:.2f}%"),
        ("  - Báo động giả (False Alarm c0)", ["accuracy_metrics", "overall", "false_alarm_c0"], "{:.2f}%"),
        ("  - Nhầm người (Confusion c0)", ["accuracy_metrics", "overall", "confusion_c0"], "{:.2f}%"),
        ("DER Thoại Thật 2 Người (c0)", ["accuracy_metrics", "slice_2spk_real", "der_collar0"], "{:.2f}%"),
        ("DER Thoại Thật 2 Người (c0.25)", ["accuracy_metrics", "slice_2spk_real", "der_collar025"], "{:.2f}%"),
        ("DER Đa Người Nói 3–7 Người (c0)", ["accuracy_metrics", "slice_3to7spk_sim", "der_collar0"], "{:.2f}%"),
        ("DER Đa Người Nói 3–7 Người (c0.25)", ["accuracy_metrics", "slice_3to7spk_sim", "der_collar025"], "{:.2f}%"),
        ("Jaccard Error Rate (JER)", ["accuracy_metrics", "overall", "jer"], "{:.2f}%"),
        ("Độ chính xác đếm người (SCA)", ["accuracy_metrics", "overall", "sca_pct"], "{:.1f}%"),
        ("Độ trễ trung bình Chunk (ms)", ["latency_metrics", "mean_chunk_latency_ms"], "{:.2f} ms"),
        ("Độ trễ phân vị p50 (ms)", ["latency_metrics", "p50_latency_ms"], "{:.2f} ms"),
        ("Độ trễ phân vị p95 (ms)", ["latency_metrics", "p95_latency_ms"], "{:.2f} ms"),
        ("Độ trễ phân vị p99 (SLA <= 320ms)", ["latency_metrics", "p99_latency_ms"], "{:.2f} ms"),
        ("Độ trễ đỉnh tối đa (Max ms)", ["latency_metrics", "max_chunk_latency_ms"], "{:.2f} ms"),
        ("Tốc độ Real-Time Factor (RTF)", ["latency_metrics", "real_time_factor_rtf"], "{:.4f}"),
        ("Đạt chuẩn SLA Streaming (p99)", ["latency_metrics", "sla_passed"], "{}")
    ]

    for label, path_keys, fmt in metrics_rows:
        v1_1_str = get_val(results.get("v1.1 (Tỷ lệ 1:3)", {}), path_keys, fmt)
        v1_3_str = get_val(results.get("v1.3 (Tỷ lệ 1:8)", {}), path_keys, fmt)
        print(f"{label:<32} | {v1_1_str:<30} | {v1_3_str:<30}")

    print("=" * 105)

    # 4. Lưu báo cáo tổng kết ra file JSON
    report_file = out_p / "streaming_benchmark_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Báo cáo đối đầu Streaming đầy đủ đã được lưu tại: {report_file}\n")


# ===============================================================================
# 5. ĐIỂM VÀO DÒNG LỆNH CLI
# ===============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá và so sánh Streaming Diarization 0.32s (v1.1 vs v1.3)")
    parser.add_argument("--model_v1_1", type=str, default=None, help="Đường dẫn checkpoint model SWA v1.1")
    parser.add_argument("--model_v1_3", type=str, default=None, help="Đường dẫn checkpoint model SWA v1.3")
    parser.add_argument("--test_manifest", type=str, default="data/manifests/test_manifest.json", help="Test manifest")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị (cuda/cpu)")
    parser.add_argument("--output_dir", type=str, default="experiments/streaming_eval", help="Thư mục xuất kết quả")
    parser.add_argument("--limit_files", type=int, default=None, help="Giới hạn số file đánh giá (để test nhanh nếu cần)")

    args = parser.parse_args()

    run_streaming_comparison(
        model_v1_1_path=args.model_v1_1,
        model_v1_3_path=args.model_v1_3,
        test_manifest=args.test_manifest,
        device=args.device,
        output_dir=args.output_dir,
        limit_files=args.limit_files
    )
