#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Module: evaluate_metrics.py
Mục đích: Đánh giá toàn diện chất lượng Diarization (DER, JER, SCA, MAE)
          theo chuẩn quốc tế (NIST / DIHARD / NOTSOFAR1) bằng pyannote.metrics.
Hỗ trợ lát cắt:
  - DER_total (Toàn bộ tập dữ liệu)
  - DER_2spk (Lát cắt 2 người nói - Dữ liệu thoại thật)
  - DER_3to7spk (Lát cắt 3-7 người nói - Dữ liệu mô phỏng)
===============================================================================
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate


def resolve_file_path(path_str: str) -> str:
    """
    Tự động chuẩn hóa và tìm kiếm file trên đa nền tảng (Windows / Linux Colab / Google Drive):
    1. Kiểm tra trực tiếp đường dẫn gốc.
    2. Nếu không tìm thấy, trích xuất đường dẫn tương đối bắt đầu từ 'data/...'
       và tìm kiếm tương đối so với thư mục làm việc hiện tại, repo root, hoặc Google Drive.
    """
    if not path_str:
        return path_str
    
    p = Path(path_str)
    if p.exists():
        return str(p)
    
    # Chuẩn hóa path phân tách bằng dấu /
    norm_path = path_str.replace("\\", "/")
    
    # Tìm kiếm đoạn path bắt đầu từ "data/"
    if "data/" in norm_path:
        sub_rel = norm_path[norm_path.find("data/"):]
        candidates = [
            Path.cwd() / sub_rel,
            Path(__file__).resolve().parent.parent.parent / sub_rel,
            Path("/content/diarization") / sub_rel,
            Path("/content/drive/MyDrive/VJ_Technologies/Nemotron_Diarization") / sub_rel,
            Path("/content/drive/MyDrive/diarization") / sub_rel,
            Path("/content/drive/MyDrive") / sub_rel,
        ]
        # Thử cả trường hợp file nằm trong javis/ trên Drive hoặc giải nén lồng thư mục data_audio_standard
        if "data/javis/" in norm_path:
            javis_sub = norm_path[norm_path.find("javis/"):]
            pure_sub = javis_sub.replace("javis/", "")
            candidates.extend([
                Path.cwd() / "data" / javis_sub,
                Path("/content/diarization/data") / javis_sub,
                Path("/content/diarization/data/javis/data_audio_standard") / pure_sub,
                Path("/content/diarization/data/data_audio_standard") / pure_sub,
                Path("/content/drive/MyDrive/VJ_Technologies/Nemotron_Diarization/data") / javis_sub,
                Path("/content/drive/MyDrive") / javis_sub,
                Path("/content/drive/MyDrive/javis") / pure_sub,
                Path("/content/drive/MyDrive/data_audio_standard") / pure_sub
            ])

        for cand in candidates:
            if cand.exists():
                return str(cand)
    
    return path_str


def parse_rttm_file(rttm_path: str, uri: Optional[str] = None) -> Annotation:
    """
    Đọc file RTTM chuẩn NIST 10 cột thành pyannote.core.Annotation.
    Định dạng dòng: SPEAKER <file_id> <chnl> <onset> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    """
    resolved_path = resolve_file_path(rttm_path)
    path_obj = Path(resolved_path)
    file_uri = uri or path_obj.stem
    annotation = Annotation(uri=file_uri)
    
    if not path_obj.exists():
        return annotation

    with open(resolved_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("SPEAKER"):
                continue
            parts = line.split()
            if len(parts) >= 8:
                try:
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker = parts[7]
                    if duration > 0.001:
                        annotation[Segment(start, start + duration)] = speaker
                except (ValueError, IndexError):
                    continue
                    
    return annotation


def extract_der_components(der_obj) -> Dict[str, float]:
    """
    Trích xuất chi tiết 3 thành phần lỗi của DER theo chuẩn NIST:
    DER = Miss Rate (Bỏ sót) + False Alarm (Báo động giả) + Speaker Confusion (Nhầm người nói).
    """
    acc = der_obj.accumulated_
    total_time = acc.get("total", 0.0)
    if total_time <= 1e-6:
        return {"der": 0.0, "miss": 0.0, "fa": 0.0, "confusion": 0.0}
    return {
        "der": round(abs(der_obj) * 100, 2),
        "miss": round((acc.get("missed detection", 0.0) / total_time) * 100, 2),
        "fa": round((acc.get("false alarm", 0.0) / total_time) * 100, 2),
        "confusion": round((acc.get("confusion", 0.0) / total_time) * 100, 2),
    }


def compute_metrics_for_dataset(
    manifest_path: str,
    hyp_rttm_dir: str,
    output_json: Optional[str] = None,
    collar_main: float = 0.0,
    collar_secondary: float = 0.25,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Chạy đánh giá toàn bộ các file trong manifest với các file hypothesis RTTM tương ứng.
    """
    manifest_p = Path(manifest_path)
    hyp_dir_p = Path(hyp_rttm_dir)

    if not manifest_p.exists():
        raise FileNotFoundError(f"Không tìm thấy manifest file: {manifest_path}")
    if not hyp_dir_p.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục hypothesis RTTM: {hyp_rttm_dir}")

    # Đọc manifest JSONL
    manifest_items = []
    with open(manifest_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_items.append(json.loads(line))

    # Khởi tạo các metric cumulative
    der_metric_c0 = DiarizationErrorRate(collar=collar_main, skip_overlap=False)
    der_metric_c025 = DiarizationErrorRate(collar=collar_secondary, skip_overlap=False)
    jer_metric = JaccardErrorRate(collar=collar_main, skip_overlap=False)

    # Sub-metrics cho từng lát cắt (Slice)
    # Lát cắt 1: 2 speakers (Real data)
    slice_2spk_c0 = DiarizationErrorRate(collar=collar_main, skip_overlap=False)
    slice_2spk_c025 = DiarizationErrorRate(collar=collar_secondary, skip_overlap=False)
    slice_2spk_jer = JaccardErrorRate(collar=collar_main, skip_overlap=False)
    
    # Lát cắt 2: 3-7 speakers (Synthetic data)
    slice_multi_c0 = DiarizationErrorRate(collar=collar_main, skip_overlap=False)
    slice_multi_c025 = DiarizationErrorRate(collar=collar_secondary, skip_overlap=False)
    slice_multi_jer = JaccardErrorRate(collar=collar_main, skip_overlap=False)

    file_results = []
    speaker_count_correct = 0
    speaker_count_abs_errors = []

    slice_2spk_correct = 0
    slice_2spk_abs_errors = []
    
    slice_multi_correct = 0
    slice_multi_abs_errors = []

    missing_hyp_count = 0

    for item in manifest_items:
        audio_path = item.get("audio_filepath", "")
        ref_rttm_path = item.get("rttm_filepath", "")
        duration = float(item.get("duration", 0.0))
        stem = Path(audio_path).stem

        hyp_rttm_path = hyp_dir_p / f"{stem}.rttm"
        if not hyp_rttm_path.exists():
            missing_hyp_count += 1
            continue

        ref_annot = parse_rttm_file(ref_rttm_path, uri=stem)
        hyp_annot = parse_rttm_file(str(hyp_rttm_path), uri=stem)

        # UEM timeline giới hạn từ 0 đến audio duration
        uem_timeline = Timeline([Segment(0.0, duration)]) if duration > 0 else None

        # Số lượng người nói
        ref_num_spk = len(ref_annot.labels())
        hyp_num_spk = len(hyp_annot.labels())
        
        is_correct_count = (ref_num_spk == hyp_num_spk)
        count_err = abs(hyp_num_spk - ref_num_spk)

        if is_correct_count:
            speaker_count_correct += 1
        speaker_count_abs_errors.append(count_err)

        # Đánh giá đơn lẻ từng file
        f_der_c0_obj = DiarizationErrorRate(collar=collar_main, skip_overlap=False)
        f_der_c0_obj(ref_annot, hyp_annot, uem=uem_timeline)

        f_der_c025_obj = DiarizationErrorRate(collar=collar_secondary, skip_overlap=False)
        f_der_c025_obj(ref_annot, hyp_annot, uem=uem_timeline)

        f_jer_obj = JaccardErrorRate(collar=collar_main, skip_overlap=False)
        f_jer = f_jer_obj(ref_annot, hyp_annot, uem=uem_timeline)

        # Cập nhật Global cumulative
        der_metric_c0(ref_annot, hyp_annot, uem=uem_timeline)
        der_metric_c025(ref_annot, hyp_annot, uem=uem_timeline)
        jer_metric(ref_annot, hyp_annot, uem=uem_timeline)

        # Phân loại lát cắt (2 người vs 3-7 người)
        is_2spk = (ref_num_spk <= 2)
        if is_2spk:
            slice_2spk_c0(ref_annot, hyp_annot, uem=uem_timeline)
            slice_2spk_c025(ref_annot, hyp_annot, uem=uem_timeline)
            slice_2spk_jer(ref_annot, hyp_annot, uem=uem_timeline)
            if is_correct_count:
                slice_2spk_correct += 1
            slice_2spk_abs_errors.append(count_err)
        else:
            slice_multi_c0(ref_annot, hyp_annot, uem=uem_timeline)
            slice_multi_c025(ref_annot, hyp_annot, uem=uem_timeline)
            slice_multi_jer(ref_annot, hyp_annot, uem=uem_timeline)
            if is_correct_count:
                slice_multi_correct += 1
            slice_multi_abs_errors.append(count_err)

        f_c0_comp = extract_der_components(f_der_c0_obj)
        f_c025_comp = extract_der_components(f_der_c025_obj)

        file_results.append({
            "stem": stem,
            "duration": duration,
            "ref_spk": ref_num_spk,
            "hyp_spk": hyp_num_spk,
            "der_collar0": f_c0_comp["der"],
            "miss_rate_c0": f_c0_comp["miss"],
            "false_alarm_c0": f_c0_comp["fa"],
            "confusion_c0": f_c0_comp["confusion"],
            "der_collar025": f_c025_comp["der"],
            "miss_rate_c025": f_c025_comp["miss"],
            "false_alarm_c025": f_c025_comp["fa"],
            "confusion_c025": f_c025_comp["confusion"],
            "jer": round(f_jer * 100, 2),
            "slice": "2spk_real" if is_2spk else "3to7spk_sim"
        })

    # Tổng kết chỉ số Global
    total_evaluated = len(file_results)
    if total_evaluated == 0:
        raise RuntimeError("Không có file nào được đánh giá! Hãy kiểm tra đường dẫn hypothesis RTTM.")

    global_c0 = extract_der_components(der_metric_c0)
    global_c025 = extract_der_components(der_metric_c025)
    global_jer = round(abs(jer_metric) * 100, 2)
    global_sca = round((speaker_count_correct / total_evaluated) * 100, 2)
    global_mae = round(float(np.mean(speaker_count_abs_errors)), 2)

    # Slice 2spk
    slice_2_count = len(slice_2spk_abs_errors)
    s2_c0 = extract_der_components(slice_2spk_c0)
    s2_c025 = extract_der_components(slice_2spk_c025)
    s2_jer = round(abs(slice_2spk_jer) * 100, 2) if slice_2_count > 0 else 0.0
    s2_sca = round((slice_2spk_correct / slice_2_count) * 100, 2) if slice_2_count > 0 else 0.0
    s2_mae = round(float(np.mean(slice_2spk_abs_errors)), 2) if slice_2_count > 0 else 0.0

    # Slice 3-7spk
    slice_m_count = len(slice_multi_abs_errors)
    sm_c0 = extract_der_components(slice_multi_c0)
    sm_c025 = extract_der_components(slice_multi_c025)
    sm_jer = round(abs(slice_multi_jer) * 100, 2) if slice_m_count > 0 else 0.0
    sm_sca = round((slice_multi_correct / slice_m_count) * 100, 2) if slice_m_count > 0 else 0.0
    sm_mae = round(float(np.mean(slice_multi_abs_errors)), 2) if slice_m_count > 0 else 0.0

    summary = {
        "manifest": str(manifest_path),
        "hyp_dir": str(hyp_rttm_dir),
        "total_files": len(manifest_items),
        "evaluated_files": total_evaluated,
        "missing_files": missing_hyp_count,
        "overall": {
            "der_collar0": global_c0["der"],
            "miss_rate_c0": global_c0["miss"],
            "false_alarm_c0": global_c0["fa"],
            "confusion_c0": global_c0["confusion"],
            "der_collar025": global_c025["der"],
            "miss_rate_c025": global_c025["miss"],
            "false_alarm_c025": global_c025["fa"],
            "confusion_c025": global_c025["confusion"],
            "jer": global_jer,
            "sca_pct": global_sca,
            "mae": global_mae
        },
        "slice_2spk_real": {
            "count": slice_2_count,
            "der_collar0": s2_c0["der"],
            "miss_rate_c0": s2_c0["miss"],
            "false_alarm_c0": s2_c0["fa"],
            "confusion_c0": s2_c0["confusion"],
            "der_collar025": s2_c025["der"],
            "miss_rate_c025": s2_c025["miss"],
            "false_alarm_c025": s2_c025["fa"],
            "confusion_c025": s2_c025["confusion"],
            "jer": s2_jer,
            "sca_pct": s2_sca,
            "mae": s2_mae
        },
        "slice_3to7spk_sim": {
            "count": slice_m_count,
            "der_collar0": sm_c0["der"],
            "miss_rate_c0": sm_c0["miss"],
            "false_alarm_c0": sm_c0["fa"],
            "confusion_c0": sm_c0["confusion"],
            "der_collar025": sm_c025["der"],
            "miss_rate_c025": sm_c025["miss"],
            "false_alarm_c025": sm_c025["fa"],
            "confusion_c025": sm_c025["confusion"],
            "jer": sm_jer,
            "sca_pct": sm_sca,
            "mae": sm_mae
        },
        "file_details": file_results
    }

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        if verbose:
            print(f"[+] Báo cáo kết quả chi tiết đã lưu tại: {out_p}")

    if verbose:
        print_benchmark_table(summary)

    return summary


def print_benchmark_table(summary: Dict[str, Any]):
    """
    In bảng tổng kết chỉ số chuẩn MLOps ra màn hình console,
    hiển thị chi tiết DER và 3 thành phần cấu thành (Miss Rate + False Alarm + Speaker Confusion).
    """
    ov = summary["overall"]
    s2 = summary["slice_2spk_real"]
    sm = summary["slice_3to7spk_sim"]

    print("\n" + "=" * 110)
    print("📊 BẢNG TỔNG KẾT KẾT QUẢ DIARIZATION BENCHMARK (CHUẨN NIST / NOTSOFAR1)")
    print("=" * 110)
    print(f"[*] Tập dữ liệu:  {summary['manifest']}")
    print(f"[*] Tổng số file: {summary['evaluated_files']} / {summary['total_files']} files")
    print("-" * 110)
    print(f"{'LÁT CẮT (SLICE)':<20} | {'FILES':<5} | {'DER (c0)':<9} | {'Miss':<7} | {'FA':<7} | {'Conf':<7} | {'DER (c0.25)':<11} | {'JER':<7} | {'SCA':<6} | {'MAE':<4}")
    print("-" * 110)
    print(f"{'1. TỔNG THỂ (Overall)':<20} | {summary['evaluated_files']:<5} | {ov['der_collar0']:>6.2f}%  | {ov.get('miss_rate_c0', 0.0):>5.2f}% | {ov.get('false_alarm_c0', 0.0):>5.2f}% | {ov.get('confusion_c0', 0.0):>5.2f}% | {ov['der_collar025']:>7.2f}%    | {ov['jer']:>5.2f}% | {ov['sca_pct']:>5.1f}% | {ov['mae']:>4.2f}")
    print(f"{'2. 2 Người (Real)':<20} | {s2['count']:<5} | {s2['der_collar0']:>6.2f}%  | {s2.get('miss_rate_c0', 0.0):>5.2f}% | {s2.get('false_alarm_c0', 0.0):>5.2f}% | {s2.get('confusion_c0', 0.0):>5.2f}% | {s2['der_collar025']:>7.2f}%    | {s2['jer']:>5.2f}% | {s2['sca_pct']:>5.1f}% | {s2['mae']:>4.2f}")
    print(f"{'3. 3-7 Người (Sim)':<20} | {sm['count']:<5} | {sm['der_collar0']:>6.2f}%  | {sm.get('miss_rate_c0', 0.0):>5.2f}% | {sm.get('false_alarm_c0', 0.0):>5.2f}% | {sm.get('confusion_c0', 0.0):>5.2f}% | {sm['der_collar025']:>7.2f}%    | {sm['jer']:>5.2f}% | {sm['sca_pct']:>5.1f}% | {sm['mae']:>4.2f}")
    print("-" * 110)
    print(f"[*] Chi tiết phân rã lỗi Collar 0.0s (DER = Miss Rate + False Alarm + Speaker Confusion):")
    print(f"    - Tổng thể:   DER = {ov['der_collar0']:>6.2f}%  =  Miss: {ov.get('miss_rate_c0', 0.0):>5.2f}% + FA: {ov.get('false_alarm_c0', 0.0):>5.2f}% + Confusion: {ov.get('confusion_c0', 0.0):>5.2f}%")
    print(f"    - 2 Người:    DER = {s2['der_collar0']:>6.2f}%  =  Miss: {s2.get('miss_rate_c0', 0.0):>5.2f}% + FA: {s2.get('false_alarm_c0', 0.0):>5.2f}% + Confusion: {s2.get('confusion_c0', 0.0):>5.2f}%")
    print(f"    - 3-7 Người:  DER = {sm['der_collar0']:>6.2f}%  =  Miss: {sm.get('miss_rate_c0', 0.0):>5.2f}% + FA: {sm.get('false_alarm_c0', 0.0):>5.2f}% + Confusion: {sm.get('confusion_c0', 0.0):>5.2f}%")
    print(f"[*] Chi tiết phân rã lỗi Collar 0.25s (Tham chiếu NIST):")
    print(f"    - Tổng thể:   DER = {ov['der_collar025']:>6.2f}%  =  Miss: {ov['miss_rate_c025']:>5.2f}% + FA: {ov['false_alarm_c025']:>5.2f}% + Confusion: {ov['confusion_c025']:>5.2f}%")
    print(f"    - 2 Người:    DER = {s2['der_collar025']:>6.2f}%  =  Miss: {s2.get('miss_rate_c025', 0.0):>5.2f}% + FA: {s2.get('false_alarm_c025', 0.0):>5.2f}% + Confusion: {s2.get('confusion_c025', 0.0):>5.2f}%")
    print(f"    - 3-7 Người:  DER = {sm['der_collar025']:>6.2f}%  =  Miss: {sm.get('miss_rate_c025', 0.0):>5.2f}% + FA: {sm.get('false_alarm_c025', 0.0):>5.2f}% + Confusion: {sm.get('confusion_c025', 0.0):>5.2f}%")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đánh giá chỉ số Speaker Diarization bằng pyannote.metrics")
    parser.add_argument("--manifest", type=str, required=True, help="Đường dẫn file manifest JSONL (ví dụ dev_manifest.json)")
    parser.add_argument("--hyp_rttm_dir", type=str, required=True, help="Thư mục chứa các file RTTM hypothesis")
    parser.add_argument("--output_json", type=str, default=None, help="Đường dẫn lưu kết quả summary dạng JSON")
    parser.add_argument("--collar_main", type=float, default=0.0, help="Collar chính (mặc định 0.0s theo NOTSOFAR1)")
    parser.add_argument("--collar_secondary", type=float, default=0.25, help="Collar phụ (mặc định 0.25s)")
    
    args = parser.parse_args()
    compute_metrics_for_dataset(
        manifest_path=args.manifest,
        hyp_rttm_dir=args.hyp_rttm_dir,
        output_json=args.output_json,
        collar_main=args.collar_main,
        collar_secondary=args.collar_secondary,
        verbose=True
    )
