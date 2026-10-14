#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script: tune_postprocessing.py (v0.2)
Mục đích: Tinh chỉnh siêu tham số Hậu xử lý (Decision Threshold, Min Speech,
          Min Silence) trên tập Dev để thiết lập mốc tham chiếu tối ưu v0.2.
Lợi thế:
  - Khai thác ma trận xác suất thô (Probabilities) đã trích xuất từ v0.1.
  - Không cần chạy lại mạng neural network nặng nề -> Tốc độ quét cực nhanh!
Không gian tìm kiếm (Grid Search):
  - threshold: [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
  - min_duration_on (Min Speech): [0.05s, 0.10s, 0.15s, 0.20s]
  - min_duration_off (Min Silence to merge): [0.05s, 0.10s, 0.15s]
Đầu ra:
  - Cấu hình tối ưu: experiments/baselines/post_proc_v0_2/best_hyperparameters.json
  - RTTM tối ưu v0.2: experiments/baselines/post_proc_v0_2/rttm/
  - Báo cáo so sánh v0.1 vs v0.2: experiments/baselines/post_proc_v0_2/metrics_report.json
===============================================================================
"""

import os
import sys
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

import numpy as np
from tqdm import tqdm
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

# Đưa thư mục hiện tại vào sys.path để import module cục bộ
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from evaluate_metrics import parse_rttm_file, compute_metrics_for_dataset


def binarize_and_filter_probs(
    probs: np.ndarray,
    frame_step: float = 0.01,
    threshold: float = 0.5,
    min_duration_on: float = 0.1,
    min_duration_off: float = 0.1
) -> List[Tuple[float, float, str]]:
    """
    Chuyển đổi ma trận xác suất (T, max_speakers) thành danh sách các đoạn thoại (start, end, speaker_id).
    Áp dụng thuật toán:
      1. Binarize: probs >= threshold
      2. Gap bridging: Nối liền các đoạn ngắt nghỉ có độ dài < min_duration_off
      3. Short speech filtering: Bỏ các đoạn thoại ngắn hơn min_duration_on
    """
    num_frames, max_speakers = probs.shape
    segments = []

    for spk_idx in range(max_speakers):
        active = (probs[:, spk_idx] >= threshold)
        if not np.any(active):
            continue

        # Tìm các khối liên tục True
        diff = np.diff(active.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1

        if active[0]:
            starts = np.r_[0, starts]
        if active[-1]:
            ends = np.r_[ends, num_frames]

        spk_segments = []
        for s, e in zip(starts, ends):
            start_sec = s * frame_step
            end_sec = e * frame_step
            if end_sec > start_sec:
                spk_segments.append([start_sec, end_sec])

        if not spk_segments:
            continue

        # BƯỚC 1: Gap bridging (Nối các khoảng lặng nhỏ hơn min_duration_off)
        merged_segments = [spk_segments[0]]
        for cur in spk_segments[1:]:
            prev = merged_segments[-1]
            gap = cur[0] - prev[1]
            if gap < min_duration_off:
                # Nối liền
                prev[1] = cur[1]
            else:
                merged_segments.append(cur)

        # BƯỚC 2: Short speech filtering (Bỏ các đoạn ngắn hơn min_duration_on)
        for seg in merged_segments:
            duration = seg[1] - seg[0]
            if duration >= min_duration_on:
                segments.append((seg[0], seg[1], f"speaker_{spk_idx}"))

    return segments


def segments_to_annotation(segments: List[Tuple[float, float, str]], uri: str) -> Annotation:
    """Chuyển danh sách segment thành pyannote.core.Annotation."""
    annot = Annotation(uri=uri)
    for start, end, spk in segments:
        annot[Segment(start, end)] = spk
    return annot


def write_segments_to_rttm(segments: List[Tuple[float, float, str]], uri: str, rttm_path: str):
    """Ghi danh sách segment ra file RTTM."""
    Path(rttm_path).parent.mkdir(parents=True, exist_ok=True)
    with open(rttm_path, "w", encoding="utf-8") as f:
        for start, end, spk in sorted(segments, key=lambda x: x[0]):
            duration = max(0.01, end - start)
            line = f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            f.write(line)


def run_postprocessing_tuning(
    manifest_path: str = "data/manifests/dev_manifest.json",
    probs_dir: str = "experiments/baselines/zero_shot_v0_1/probabilities",
    output_dir: str = "experiments/baselines/post_proc_v0_2",
    v0_1_report_path: str = "experiments/baselines/zero_shot_v0_1/metrics_report.json"
):
    """
    Quy trình Grid Search Post-processing:
      1. Đọc dữ liệu probabilities của toàn bộ tập Dev từ bước v0.1
      2. Quét ma trận (threshold, min_duration_on, min_duration_off)
      3. Tìm bộ tham số có DER (Collar 0s) thấp nhất
      4. Xuất RTTM tối ưu và báo cáo so sánh v0.1 vs v0.2
    """
    manifest_p = Path(manifest_path)
    probs_dir_p = Path(probs_dir)
    out_dir_p = Path(output_dir)
    rttm_out_dir = out_dir_p / "rttm"
    rttm_out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("🎯 BẮT ĐẦU TINH CHỈNH HẬU XỬ LÝ (POST-PROCESSING TUNING v0.2)")
    print("=" * 80)
    print(f"[*] Manifest:          {manifest_path}")
    print(f"[*] Probabilities Dir: {probs_dir}")
    print(f"[*] Output Dir:        {output_dir}")

    if not probs_dir_p.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục xác suất {probs_dir}! "
            f"Vui lòng chạy script run_zero_shot_baseline.py (v0.1) trước."
        )

    # Đọc manifest
    manifest_items = []
    with open(manifest_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_items.append(json.loads(line))

    # Nạp toàn bộ dữ liệu xác suất và reference annotations vào RAM
    print("\n[*] Đang nạp trước ma trận probabilities và ground truth RTTM vào bộ nhớ...")
    cache_data = []
    for item in manifest_items:
        audio_path = item["audio_filepath"]
        ref_rttm = item["rttm_filepath"]
        duration = float(item.get("duration", 0.0))
        stem = Path(audio_path).stem
        prob_file = probs_dir_p / f"{stem}.npz"

        if not prob_file.exists():
            continue

        npz = np.load(prob_file)
        probs = npz["probs"]
        frame_step = float(npz.get("frame_step", 0.01))
        ref_annot = parse_rttm_file(ref_rttm, uri=stem)
        uem = Timeline([Segment(0.0, duration)]) if duration > 0 else None

        cache_data.append({
            "stem": stem,
            "probs": probs,
            "frame_step": frame_step,
            "ref_annot": ref_annot,
            "uem": uem
        })

    total_cached = len(cache_data)
    print(f"[+] Đã nạp thành công {total_cached} / {len(manifest_items)} files vào RAM.")
    if total_cached == 0:
        raise RuntimeError("Không có file xác suất nào để tinh chỉnh!")

    # Thiết lập Grid Search
    threshold_candidates = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    min_on_candidates = [0.05, 0.10, 0.15, 0.20]
    min_off_candidates = [0.05, 0.10, 0.15]

    total_combinations = len(threshold_candidates) * len(min_on_candidates) * len(min_off_candidates)
    print(f"\n[*] Tổng số bộ tham số cần quét: {total_combinations} cấu hình...")
    print(f"    - Threshold:       {threshold_candidates}")
    print(f"    - Min Speech (on): {min_on_candidates}")
    print(f"    - Min Gap (off):   {min_off_candidates}")

    best_der = float("inf")
    best_params = None
    results_grid = []

    pbar = tqdm(total=total_combinations, desc="Grid Search Tuning", unit="cfg")

    for th in threshold_candidates:
        for m_on in min_on_candidates:
            for m_off in min_off_candidates:
                der_eval = DiarizationErrorRate(collar=0.0, skip_overlap=False)
                for item in cache_data:
                    segs = binarize_and_filter_probs(
                        item["probs"],
                        frame_step=item["frame_step"],
                        threshold=th,
                        min_duration_on=m_on,
                        min_duration_off=m_off
                    )
                    hyp_annot = segments_to_annotation(segs, uri=item["stem"])
                    der_eval(item["ref_annot"], hyp_annot, uem=item["uem"])

                current_der = abs(der_eval) * 100
                results_grid.append({
                    "threshold": th,
                    "min_duration_on": m_on,
                    "min_duration_off": m_off,
                    "der_collar0": round(current_der, 2)
                })

                if current_der < best_der:
                    best_der = current_der
                    best_params = {
                        "threshold": th,
                        "min_duration_on": m_on,
                        "min_duration_off": m_off,
                        "best_dev_der_collar0": round(best_der, 2)
                    }

                pbar.update(1)
                pbar.set_postfix({"Best DER": f"{best_der:.2f}%", "Cur": f"{current_der:.2f}%"})

    pbar.close()

    print("\n" + "=" * 80)
    print("🏆 BỘ THAM SỐ HẬU XỬ LÝ TỐI ƯU NHẤT TRÊN TẬP DEV (v0.2):")
    print("=" * 80)
    print(f"  [+] Ngưỡng quyết định (Decision Threshold):    {best_params['threshold']}")
    print(f"  [+] Thời lượng nói tối thiểu (Min Speech On):   {best_params['min_duration_on']}s")
    print(f"  [+] Khoảng ngắt nghỉ nối liền (Min Gap Off):    {best_params['min_duration_off']}s")
    print(f"  [+] Điểm DER tập Dev (Collar 0s):               {best_params['best_dev_der_collar0']}%")
    print("=" * 80)

    # Lưu file config tốt nhất
    best_config_file = out_dir_p / "best_hyperparameters.json"
    with open(best_config_file, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print(f"[+] Cấu hình tối ưu đã lưu tại: {best_config_file}")

    # Tạo RTTM tối ưu cho toàn bộ file và xuất ra rttm/
    print("\n[*] Đang tạo các file RTTM tối ưu v0.2 cho toàn bộ tập Dev...")
    for item in cache_data:
        segs = binarize_and_filter_probs(
            item["probs"],
            frame_step=item["frame_step"],
            threshold=best_params["threshold"],
            min_duration_on=best_params["min_duration_on"],
            min_duration_off=best_params["min_duration_off"]
        )
        out_rttm = rttm_out_dir / f"{item['stem']}.rttm"
        write_segments_to_rttm(segs, uri=item["stem"], rttm_path=str(out_rttm))

    # Đánh giá toàn diện các lát cắt cho v0.2
    v0_2_report_file = out_dir_p / "metrics_report.json"
    v0_2_summary = compute_metrics_for_dataset(
        manifest_path=manifest_path,
        hyp_rttm_dir=str(rttm_out_dir),
        output_json=str(v0_2_report_file),
        collar_main=0.0,
        collar_secondary=0.25,
        verbose=True
    )

    # In bảng so sánh đối đầu v0.1 vs v0.2
    if Path(v0_1_report_path).exists():
        with open(v0_1_report_path, "r", encoding="utf-8") as f:
            v0_1_summary = json.load(f)
        print_comparison_table(v0_1_summary, v0_2_summary)

    return best_params, v0_2_summary


def print_comparison_table(v0_1: Dict[str, Any], v0_2: Dict[str, Any]):
    """
    In bảng so sánh đối đầu giữa Baseline Zero-shot v0.1 và Post-processed v0.2.
    """
    ov1 = v0_1["overall"]
    ov2 = v0_2["overall"]
    s2_1 = v0_1["slice_2spk_real"]
    s2_2 = v0_2["slice_2spk_real"]
    sm_1 = v0_1["slice_3to7spk_sim"]
    sm_2 = v0_2["slice_3to7spk_sim"]

    der_diff = ov2["der_collar0"] - ov1["der_collar0"]
    s2_diff = s2_2["der_collar0"] - s2_1["der_collar0"]
    sm_diff = sm_2["der_collar0"] - sm_1["der_collar0"]

    print("\n" + "=" * 80)
    print("📈 SO SÁNH ĐỐI ĐẦU: BASELINE ZERO-SHOT (v0.1) VS POST-PROCESSED (v0.2)")
    print("=" * 80)
    print(f"{'CHỈ SỐ ĐÁNH GIÁ':<30} | {'v0.1 (Zero-shot)':<18} | {'v0.2 (Post-proc)':<18} | {'CẢI THIỆN (Δ)':<12}")
    print("-" * 80)
    print(f"{'Dev DER tổng (Collar 0s)':<30} | {ov1['der_collar0']:>6.2f}%            | {ov2['der_collar0']:>6.2f}%            | {der_diff:>+6.2f}%")
    print(f"{'Dev DER 2 người (Real)':<30} | {s2_1['der_collar0']:>6.2f}%            | {s2_2['der_collar0']:>6.2f}%            | {s2_diff:>+6.2f}%")
    print(f"{'Dev DER 3-7 người (Sim)':<30} | {sm_1['der_collar0']:>6.2f}%            | {sm_2['der_collar0']:>6.2f}%            | {sm_diff:>+6.2f}%")
    print(f"{'Dev DER (Collar 0.25s)':<30} | {ov1['der_collar025']:>6.2f}%            | {ov2['der_collar025']:>6.2f}%            | {ov2['der_collar025'] - ov1['der_collar025']:>+6.2f}%")
    print(f"{'Jaccard Error Rate (JER)':<30} | {ov1['jer']:>6.2f}%            | {ov2['jer']:>6.2f}%            | {ov2['jer'] - ov1['jer']:>+6.2f}%")
    print(f"{'Speaker Counting Acc (SCA)':<30} | {ov1['sca_pct']:>6.1f}%            | {ov2['sca_pct']:>6.1f}%            | {ov2['sca_pct'] - ov1['sca_pct']:>+6.1f}%")
    print(f"{'Speaker Counting MAE':<30} | {ov1['mae']:>6.2f}             | {ov2['mae']:>6.2f}             | {ov2['mae'] - ov1['mae']:>+6.2f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tinh chỉnh Hậu xử lý v0.2 cho Nemotron Diarization")
    parser.add_argument("--manifest", type=str, default="data/manifests/dev_manifest.json", help="Đường dẫn dev_manifest.json")
    parser.add_argument("--probs_dir", type=str, default="experiments/baselines/zero_shot_v0_1/probabilities", help="Thư mục chứa file xác suất thô .npz")
    parser.add_argument("--output_dir", type=str, default="experiments/baselines/post_proc_v0_2", help="Thư mục xuất kết quả v0.2")
    parser.add_argument("--v0_1_report", type=str, default="experiments/baselines/zero_shot_v0_1/metrics_report.json", help="Báo cáo v0.1 để so sánh")

    args = parser.parse_args()
    run_postprocessing_tuning(
        manifest_path=args.manifest,
        probs_dir=args.probs_dir,
        output_dir=args.output_dir,
        v0_1_report_path=args.v0_1_report
    )
