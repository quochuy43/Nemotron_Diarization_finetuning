#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script: run_zero_shot_baseline.py (v0.1)
Mục đích: Chạy suy luận Streaming Ultra-Low Latency (0.32s) của mô hình
          NVIDIA Nemotron-3-Diarization trên tập Dev (43 files - 6.16h).
Đặc tả kỹ thuật (Ultra-Low Latency 0.32s):
  - chunk_len = 3 (240ms)
  - chunk_right_context = 1 (80ms)
  - fifo_len = 264
  - spkcache_len = 264
  - spkcache_update_period = 222
  - Độ trễ đệm buffer: (3 + 1) * 80ms = 320ms = 0.32s
Đầu ra:
  - RTTM dự đoán: experiments/baselines/zero_shot_v0_1/rttm/
  - Xác suất thô (Frame probs): experiments/baselines/zero_shot_v0_1/probabilities/
  - Báo cáo chỉ số (DER / JER / SCA / MAE / RTF): experiments/baselines/zero_shot_v0_1/metrics_report.json
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

# Đưa thư mục hiện tại vào sys.path để import module cục bộ
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from evaluate_metrics import compute_metrics_for_dataset


# Cấu hình chuẩn Ultra-Low Latency 0.32s từ NVIDIA
STREAMING_CONFIG = {
    "chunk_len": 3,
    "chunk_right_context": 1,
    "fifo_len": 264,
    "spkcache_len": 264,
    "spkcache_update_period": 222,
    "latency_seconds": 0.32,
}


def check_and_get_backend() -> str:
    """
    Tự động phát hiện backend khả dụng (transformers hoặc nemo).
    Transformers 5.19+ hỗ trợ trực tiếp và ổn định nhất cho Nemotron-3 RoPE architecture trên Windows.
    """
    has_transformers = False
    has_nemo = False

    try:
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        has_transformers = True
    except Exception:
        has_transformers = False

    try:
        import nemo.collections.asr.models as asr_models
        has_nemo = True
    except Exception:
        has_nemo = False

    if has_transformers:
        return "transformers"
    elif has_nemo:
        return "nemo"
    else:
        print("\n" + "!" * 80)
        print("❌ CẢNH BÁO: Chưa tìm thấy backend phù hợp (Transformers hoặc NeMo tương thích).")
        print("!" * 80 + "\n")
        return "none"


class NeMoDiarizationRunner:
    """Runner suy luận sử dụng NVIDIA NeMo SortformerEncLabelModel."""
    def __init__(self, model_name_or_path: str = "nvidia/Nemotron-3-Diarization", device: str = "cuda"):
        from nemo.collections.asr.models import SortformerEncLabelModel
        print(f"[*] Đang nạp mô hình qua NeMo Backend: {model_name_or_path}...")
        
        path_obj = Path(model_name_or_path)
        if path_obj.exists() and path_obj.suffix == ".nemo":
            self.model = SortformerEncLabelModel.restore_from(
                restore_path=str(path_obj),
                map_location=device,
                strict=False
            )
        else:
            self.model = SortformerEncLabelModel.from_pretrained(model_name_or_path)
            
        self.model = self.model.to(device)
        self.model.eval()

        # Thiết lập cấu hình streaming Ultra-Low Latency (0.32s)
        self.model.sortformer_modules.chunk_len = STREAMING_CONFIG["chunk_len"]
        self.model.sortformer_modules.chunk_right_context = STREAMING_CONFIG["chunk_right_context"]
        self.model.sortformer_modules.fifo_len = STREAMING_CONFIG["fifo_len"]
        self.model.sortformer_modules.spkcache_len = STREAMING_CONFIG["spkcache_len"]
        self.model.sortformer_modules.spkcache_update_period = STREAMING_CONFIG["spkcache_update_period"]
        self.model._check_streaming_parameters()
        print(f"[+] Đã kích hoạt cấu hình Ultra-Low Latency: buffer {STREAMING_CONFIG['latency_seconds']}s.")

    def diarize_audio(self, audio_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Thực hiện diarization trên 1 file âm thanh.
        Trả về danh sách segment và ma trận xác suất khung thời gian (frame_probs).
        """
        # NeMo diarize với include_tensor_outputs để lưu frame probabilities
        outputs = self.model.diarize(
            audio=audio_path,
            batch_size=1,
            include_tensor_outputs=True
        )
        # outputs có dạng (predicted_segments, predicted_probs)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            pred_segments, pred_probs = outputs
        else:
            pred_segments = outputs
            pred_probs = None

        segments = []
        if pred_segments and len(pred_segments) > 0:
            for item in pred_segments[0]:
                if isinstance(item, (list, tuple)) and len(item) >= 3:
                    segments.append({
                        "start": float(item[0]),
                        "end": float(item[1]),
                        "speaker": str(item[2])
                    })
                elif isinstance(item, dict):
                    segments.append({
                        "start": float(item.get("start", item.get("Start", 0.0))),
                        "end": float(item.get("end", item.get("End", 0.0))),
                        "speaker": str(item.get("speaker", item.get("Speaker", "0")))
                    })

        probs_np = None
        if pred_probs is not None:
            if isinstance(pred_probs, torch.Tensor):
                probs_np = pred_probs.detach().cpu().numpy()
            elif isinstance(pred_probs, np.ndarray):
                probs_np = pred_probs
            if probs_np is not None and probs_np.ndim == 3:
                probs_np = probs_np[0]

        return segments, probs_np


class TransformersDiarizationRunner:
    """Runner suy luận sử dụng Hugging Face Transformers."""
    def __init__(self, model_id: str = "nvidia/Nemotron-3-Diarization", device: str = "cuda"):
        from transformers import AutoModelForAudioFrameClassification, AutoProcessor
        print(f"[*] Đang nạp mô hình qua Transformers Backend: {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForAudioFrameClassification.from_pretrained(model_id).to(device)
        self.device = torch.device(device)
        self.processor.set_streaming_mode("ultra_low_latency")
        print(f"[+] Đã cấu hình streaming mode: ultra_low_latency ({self.processor.streaming_latency_ms} ms)")

    def diarize_audio(self, audio_path: str) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Thực hiện diarization offline hoặc chunked inference với transformers.
        """
        wav_data, sr = sf.read(audio_path)
        if wav_data.ndim > 1:
            wav_data = wav_data[:, 0]
        if sr != 16000:
            import torchaudio.functional as F
            wav_tensor = torch.from_numpy(wav_data).float()
            wav_tensor = F.resample(wav_tensor, sr, 16000)
            wav_data = wav_tensor.numpy()
            sr = 16000

        inputs = self.processor(wav_data, sampling_rate=16000).to(self.device, dtype=self.model.dtype)
        with torch.inference_mode():
            logits = self.model(**inputs).logits # (1, num_frames, 8)
            probs = torch.sigmoid(logits)

        raw_segments = self.processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
        segments = []
        for seg in raw_segments:
            segments.append({
                "start": float(seg["Start"]),
                "end": float(seg["End"]),
                "speaker": str(seg["Speaker"])
            })

        probs_np = probs[0].detach().cpu().numpy()
        return segments, probs_np


def segments_to_rttm(segments: List[Dict[str, Any]], uri: str, rttm_path: str):
    """
    Lưu danh sách phân đoạn sang file RTTM chuẩn NIST 10 cột.
    """
    Path(rttm_path).parent.mkdir(parents=True, exist_ok=True)
    with open(rttm_path, "w", encoding="utf-8") as f:
        for seg in sorted(segments, key=lambda x: x["start"]):
            start = seg["start"]
            duration = max(0.01, seg["end"] - seg["start"])
            spk = seg["speaker"]
            # Chuẩn hóa tên speaker thành spk_X
            if not spk.startswith("speaker_") and not spk.startswith("spk_"):
                spk = f"speaker_{spk}"
            line = f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            f.write(line)


def run_zero_shot_evaluation(
    manifest_path: str = "data/manifests/dev_manifest.json",
    output_dir: str = "experiments/baselines/zero_shot_v0_1",
    model_name_or_path: str = "nvidia/Nemotron-3-Diarization",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    limit_files: Optional[int] = None
):
    """
    Quy trình đánh giá hoàn chỉnh v0.1:
      1. Đọc danh sách file từ dev_manifest.json
      2. Chạy streaming inference 0.32s trên từng file
      3. Đo Real-Time Factor (RTF) và Latency
      4. Lưu RTTM dự đoán và file xác suất
      5. Tính toán toàn diện DER, JER, SCA, MAE bằng evaluate_metrics.py
    """
    manifest_p = Path(manifest_path)
    out_dir_p = Path(output_dir)
    rttm_out_dir = out_dir_p / "rttm"
    probs_out_dir = out_dir_p / "probabilities"
    rttm_out_dir.mkdir(parents=True, exist_ok=True)
    probs_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 BẮT ĐẦU CHẠY BASELINE ZERO-SHOT (v0.1) - NEMOTRON 0.32S")
    print("=" * 80)
    print(f"[*] Manifest:    {manifest_path}")
    print(f"[*] Device:      {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"[*] Model:       {model_name_or_path}")
    print(f"[*] Thư mục RTTM:{rttm_out_dir}")
    print(f"[*] Thư mục Probs:{probs_out_dir}")

    # Đọc manifest
    manifest_items = []
    with open(manifest_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_items.append(json.loads(line))

    if limit_files and limit_files > 0:
        print(f"[!] Chế độ debug/test nhanh: giới hạn {limit_files} files.")
        manifest_items = manifest_items[:limit_files]

    total_files = len(manifest_items)
    print(f"[*] Tổng số file cần xử lý: {total_files} files")

    # Chọn backend
    backend_type = check_and_get_backend()
    if backend_type == "none":
        sys.exit(1)

    # Khởi tạo Runner
    if backend_type == "nemo":
        runner = NeMoDiarizationRunner(model_name_or_path=model_name_or_path, device=device)
    else:
        runner = TransformersDiarizationRunner(model_id=model_name_or_path, device=device)

    total_audio_duration = 0.0
    total_compute_time = 0.0
    processed_count = 0

    print("\n--- Đang thực hiện suy luận Diarization từng file ---")
    pbar = tqdm(manifest_items, desc="Đang xử lý Dev Set", unit="file")

    for item in pbar:
        audio_path = item["audio_filepath"]
        duration = float(item.get("duration", 0.0))
        stem = Path(audio_path).stem
        out_rttm_file = rttm_out_dir / f"{stem}.rttm"
        out_prob_file = probs_out_dir / f"{stem}.npz"

        if not Path(audio_path).exists():
            print(f"[!] Bỏ qua file không tồn tại: {audio_path}")
            continue

        # Chạy suy luận và đo thời gian thực thi
        start_t = time.perf_counter()
        try:
            segments, probs = runner.diarize_audio(audio_path)
        except Exception as e:
            print(f"\n[!] Lỗi khi xử lý file {stem}: {e}")
            continue
        elapsed_t = time.perf_counter() - start_t

        # Ghi file RTTM
        segments_to_rttm(segments, uri=stem, rttm_path=str(out_rttm_file))

        # Lưu file xác suất nếu có
        if probs is not None:
            np.savez_compressed(
                out_prob_file,
                probs=probs.astype(np.float32),
                frame_step=0.01 # 10ms frame resolution
            )

        total_audio_duration += duration
        total_compute_time += elapsed_t
        processed_count += 1

        rtf = elapsed_t / duration if duration > 0 else 0.0
        pbar.set_postfix({"File": stem[:15], "RTF": f"{rtf:.3f}", "Spk": len(set(s['speaker'] for s in segments))})

    global_rtf = total_compute_time / total_audio_duration if total_audio_duration > 0 else 0.0
    print("\n" + "-" * 80)
    print(f"[*] Hoàn tất suy luận: {processed_count} / {total_files} files")
    print(f"[*] Tổng thời lượng âm thanh: {total_audio_duration / 3600:.2f} giờ")
    print(f"[*] Tổng thời gian tính toán: {total_compute_time:.2f} giây")
    print(f"[*] Real-Time Factor (RTF):  {global_rtf:.4f} (Càng nhỏ hơn 1.0 càng nhanh)")
    print("-" * 80)

    # Đánh giá chỉ số chất lượng Diarization
    print("\n--- Đang tính toán các chỉ số Diarization Error Rate (DER) & JER ---")
    metrics_json_path = out_dir_p / "metrics_report.json"
    summary = compute_metrics_for_dataset(
        manifest_path=manifest_path,
        hyp_rttm_dir=str(rttm_out_dir),
        output_json=str(metrics_json_path),
        collar_main=0.0,
        collar_secondary=0.25,
        verbose=True
    )

    # Bổ sung thông số RTF vào báo cáo
    summary["rtf"] = round(global_rtf, 4)
    summary["total_audio_duration_seconds"] = round(total_audio_duration, 2)
    summary["total_compute_time_seconds"] = round(total_compute_time, 2)
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[+] Báo cáo kết quả v0.1 đã được cập nhật tại: {metrics_json_path}")
    print("=" * 80 + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy Baseline Zero-shot v0.1 cho Nemotron-3-Diarization")
    parser.add_argument("--manifest", type=str, default="data/manifests/dev_manifest.json", help="Đường dẫn dev_manifest.json")
    parser.add_argument("--output_dir", type=str, default="experiments/baselines/zero_shot_v0_1", help="Thư mục xuất kết quả")
    parser.add_argument("--model", type=str, default="nvidia/Nemotron-3-Diarization", help="Tên model HF hoặc file .nemo")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị thực thi (cuda/cpu)")
    parser.add_argument("--limit", type=int, default=None, help="Chạy thử nghiệm giới hạn số file")

    args = parser.parse_args()
    run_zero_shot_evaluation(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_name_or_path=args.model,
        device=args.device,
        limit_files=args.limit
    )
