#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script phân chia 171 file dữ liệu thật thành Train / Dev / Test 
và tự động tạo các tập Manifest chuẩn NVIDIA NeMo Diarization theo đúng Roadmap.

Tác giả: AI Pair Programmer (Dự án Realtime Speaker Diarization Nhật Bản)
"""

import os
import glob
import json
import shutil
import argparse
import soundfile as sf
from typing import List, Dict, Tuple


def get_audio_info(wav_path: str, rttm_path: str) -> Dict:
    """Đọc thông tin thời lượng và số lượng speaker từ file WAV & RTTM."""
    info = sf.info(wav_path)
    duration = round(info.duration, 2)
    
    speakers = set()
    if os.path.exists(rttm_path):
        with open(rttm_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    speakers.add(parts[7])
    
    num_spks = len(speakers) if speakers else 2
    
    # Định dạng chuẩn đường dẫn dùng dấu gạch xuôi để tương thích đa nền tảng
    norm_wav = os.path.abspath(wav_path).replace("\\", "/")
    norm_rttm = os.path.abspath(rttm_path).replace("\\", "/")
    
    return {
        "audio_filepath": norm_wav,
        "offset": 0.0,
        "duration": duration,
        "label": "infer",
        "text": "-",
        "num_speakers": num_spks,
        "rttm_filepath": norm_rttm,
        "uem_filepath": None
    }


def split_real_dataset(real_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Phân chia 171 file theo phương pháp Phân tầng (Stratified Hold-out):
    - Train Thật (125 files ~ 26.0h): 100 CallHome + 25 CallFriend
    - Dev Set    (23 files ~ 5.1h):  10 CallHome + 3 CallFriend + 5 Themed + 5 Travel
    - Test Set   (23 files ~ 4.4h):  10 CallHome + 3 CallFriend + 5 Themed + 5 Travel
    """
    ch_wavs = sorted(glob.glob(os.path.join(real_dir, "callhome_jpn", "audio", "*.wav")))
    cf_wavs = sorted(glob.glob(os.path.join(real_dir, "callfriend_jpn", "audio", "*.wav")))
    it_wavs = sorted(glob.glob(os.path.join(real_dir, "ingcrowd_japanese_themed_dialogue", "audio", "*.wav")))
    ir_wavs = sorted(glob.glob(os.path.join(real_dir, "ingcrowd_japanese_roleplay_travel_agency", "audio", "*.wav")))

    def to_pairs(wav_list):
        pairs = []
        for w in wav_list:
            # Tìm file rttm tương ứng
            base = os.path.splitext(os.path.basename(w))[0]
            d_dir = os.path.dirname(os.path.dirname(w))
            r = os.path.join(d_dir, "rttm", f"{base}.rttm")
            if not os.path.exists(r):
                raise FileNotFoundError(f"Không tìm thấy RTTM tương ứng cho: {w}")
            pairs.append((w, r))
        return pairs

    ch_pairs = to_pairs(ch_wavs)
    cf_pairs = to_pairs(cf_wavs)
    it_pairs = to_pairs(it_wavs)
    ir_pairs = to_pairs(ir_wavs)

    # 1. Train thật: 100 CallHome + 25 CallFriend
    train_pairs = ch_pairs[:100] + cf_pairs[:25]

    # 2. Dev Set: 10 CallHome + 3 CallFriend + 5 Themed + 5 Travel
    dev_pairs = ch_pairs[100:110] + cf_pairs[25:28] + it_pairs[:5] + ir_pairs[:5]

    # 3. Test Set (Đóng băng): 10 CallHome + 3 CallFriend + 5 Themed + 5 Travel
    test_pairs = ch_pairs[110:120] + cf_pairs[28:31] + it_pairs[5:] + ir_pairs[5:]

    return train_pairs, dev_pairs, test_pairs


def write_manifest(pairs: List[Tuple[str, str]], output_manifest: str) -> List[Dict]:
    """Tạo file manifest JSON (mỗi dòng 1 JSON theo chuẩn NeMo)."""
    entries = []
    with open(output_manifest, "w", encoding="utf-8") as f:
        for wav, rttm in pairs:
            item = get_audio_info(wav, rttm)
            entries.append(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return entries


def normalize_entry_paths(item: Dict) -> Dict:
    """
    Chuẩn hóa đường dẫn: 
    1. Dùng dấu gạch xuôi '/' để tương thích 100% đa nền tảng (Windows, Linux, Colab, Docker).
    2. Tự động sửa lỗi thư mục nếu từng lưu thành synthetic_78h.
    """
    for key in ["audio_filepath", "rttm_filepath"]:
        val = item.get(key)
        if val:
            p = str(val).replace("\\", "/")
            if "synthetic_78h" in p:
                p_candidate = p.replace("synthetic_78h", "synthetic_78")
                if os.path.exists(p_candidate):
                    p = p_candidate
            item[key] = p
    return item


def load_manifest(manifest_path: str) -> List[Dict]:
    """Đọc file manifest jsonl (hỗ trợ cả single-line jsonl lẫn multi-line formatted JSON)."""
    if not os.path.exists(manifest_path):
        return []
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    # Cách 1: Thử đọc theo chuẩn JSON Lines (mỗi dòng 1 JSON)
    lines = content.splitlines()
    entries = []
    is_jsonl = True
    for line in lines:
        s = line.strip()
        if not s:
            continue
        try:
            entries.append(json.loads(s))
        except Exception:
            is_jsonl = False
            break

    if not is_jsonl or not entries:
        # Cách 2: Nếu là multi-line pretty-printed objects ({ ... }\n{ ... })
        entries = []
        decoder = json.JSONDecoder()
        idx = 0
        length = len(content)
        while idx < length:
            while idx < length and content[idx].isspace():
                idx += 1
            if idx >= length:
                break
            obj, end_idx = decoder.raw_decode(content, idx)
            entries.append(obj)
            idx = end_idx

    # Chuẩn hóa toàn bộ đường dẫn dùng dấu '/' và sửa lỗi synthetic_78h nếu có
    entries = [normalize_entry_paths(e) for e in entries]

    # Lưu lại file nguồn với đường dẫn sạch sẽ
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return entries


def main():
    parser = argparse.ArgumentParser(description="Phân chia 171 file thật & tạo các Manifest chuẩn NeMo")
    parser.add_argument("--real_dir", type=str, default="real_data", help="Đường dẫn thư mục real_data")
    parser.add_argument("--output_dir", type=str, default="data_splits", help="Thư mục lưu manifests và splits")
    parser.add_argument("--copy_files", action="store_true", help="Nếu bật, copy file vật lý vào data_splits/{train,dev,test}")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("🚀 BẮT ĐẦU PHÂN CHIA DỮ LIỆU THẬT & TẠO MANIFEST NEMO")
    print("=" * 70)

    train_pairs, dev_pairs, test_pairs = split_real_dataset(args.real_dir)

    print(f"[*] Phân chia hoàn tất:")
    print(f"    - Train (Thật): {len(train_pairs)} files")
    print(f"    - Dev Set:      {len(dev_pairs)} files (13 viễn thông + 10 Zoom)")
    print(f"    - Test Set:     {len(test_pairs)} files (13 viễn thông + 10 Zoom)")

    # 1. Lưu danh sách tên file .txt
    def save_list(pairs, filename):
        path = os.path.join(args.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for w, _ in pairs:
                f.write(f"{os.path.basename(w)}\n")

    save_list(train_pairs, "train_real_files.txt")
    save_list(dev_pairs, "dev_files.txt")
    save_list(test_pairs, "test_files.txt")

    # 2. Tạo manifest cho các tập
    print("\n[*] Đang tạo file manifest JSON chuẩn NeMo...")
    dev_manifest = os.path.join(args.output_dir, "dev_manifest.json")
    test_manifest = os.path.join(args.output_dir, "test_manifest.json")
    train_real_manifest = os.path.join(args.output_dir, "train_real_manifest.json")

    # Đọc các file synthetic từ synthetic_78 để trích xuất case 3-7 người cho Dev và Test
    sim_78_manifest = "synthetic_78/manifest.json"
    sim_78_entries = load_manifest(sim_78_manifest)

    dev_sim_entries = []
    test_sim_entries = []
    excluded_basenames = set()

    if sim_78_entries:
        multi_spk_entries = [e for e in sim_78_entries if e["num_speakers"] >= 3]
        # Dev: 10 file (3-4 người) + 10 file (5-7 người)
        dev_sim_3_4 = [e for e in multi_spk_entries if 3 <= e["num_speakers"] <= 4][:10]
        dev_sim_5_7 = [e for e in multi_spk_entries if 5 <= e["num_speakers"] <= 7][:10]
        dev_sim_entries = dev_sim_3_4 + dev_sim_5_7
        dev_paths = set(e["audio_filepath"] for e in dev_sim_entries)

        # Test: 10 file (3-4 người) + 10 file (5-7 người)
        rem_multi = [e for e in multi_spk_entries if e["audio_filepath"] not in dev_paths]
        test_sim_3_4 = [e for e in rem_multi if 3 <= e["num_speakers"] <= 4][:10]
        test_sim_5_7 = [e for e in rem_multi if 5 <= e["num_speakers"] <= 7][:10]
        test_sim_entries = test_sim_3_4 + test_sim_5_7

        excluded_basenames = set(os.path.basename(e["audio_filepath"]) for e in dev_sim_entries + test_sim_entries)
        print(f"    [+] Đã trích {len(dev_sim_entries)} file 3-7 người cho Dev và {len(test_sim_entries)} file 3-7 người cho Test (Khóa hoàn toàn khỏi Train)")

    # Gộp Thật + Mô phỏng 3-7 người vào Dev và Test
    final_dev_entries = [get_audio_info(w, r) for w, r in dev_pairs] + dev_sim_entries
    final_test_entries = [get_audio_info(w, r) for w, r in test_pairs] + test_sim_entries
    train_real_entries = [get_audio_info(w, r) for w, r in train_pairs]

    with open(dev_manifest, "w", encoding="utf-8") as f:
        for item in final_dev_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(test_manifest, "w", encoding="utf-8") as f:
        for item in final_test_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(train_real_manifest, "w", encoding="utf-8") as f:
        for item in train_real_entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    dev_dur = sum(e["duration"] for e in final_dev_entries) / 3600.0
    test_dur = sum(e["duration"] for e in final_test_entries) / 3600.0
    train_real_dur = sum(e["duration"] for e in train_real_entries) / 3600.0

    print(f"    [+] {dev_manifest} ({len(final_dev_entries)} files - {dev_dur:.2f}h | gồm 23 file Thật + {len(dev_sim_entries)} file Mô phỏng 3-7 người)")
    print(f"    [+] {test_manifest} ({len(final_test_entries)} files - {test_dur:.2f}h | gồm 23 file Thật + {len(test_sim_entries)} file Mô phỏng 3-7 người)")
    print(f"    [+] {train_real_manifest} ({len(train_real_entries)} files - {train_real_dur:.2f}h)")

    # 3. Ghép nối với dữ liệu mô phỏng theo 4 Version thí nghiệm (Đã loại bỏ 40 file Dev/Test)
    print("\n[*] Đang ghép Manifest huấn luyện cho 4 phiên bản thử nghiệm (Đã lọc Zero Data Leakage)...")

    sim_130_manifest = "synthetic_130/manifest.json"
    sim_208_manifest = "synthetic_208/manifest.json"

    sim_130_entries = load_manifest(sim_130_manifest)
    sim_208_entries = load_manifest(sim_208_manifest)

    # Hàm lọc loại bỏ các file đã dùng cho Dev / Test
    def filter_train_sim(entries):
        return [e for e in entries if os.path.basename(e["audio_filepath"]) not in excluded_basenames]

    train_sim_78 = filter_train_sim(sim_78_entries)
    train_sim_130 = filter_train_sim(sim_130_entries)
    train_sim_208 = filter_train_sim(sim_208_entries)

    # v1.0: Synthetic only (78h)
    if train_sim_78:
        p_v10 = os.path.join(args.output_dir, "train_v1_0_sim_only.json")
        with open(p_v10, "w", encoding="utf-8") as f:
            for item in train_sim_78:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        dur_v10 = sum(e["duration"] for e in train_sim_78) / 3600.0
        print(f"    [+] v1.0 (Synthetic Only): {p_v10} ({len(train_sim_78)} files - {dur_v10:.1f}h)")

    # v1.1: Ratio 1:3 (26h Thật + 78h Phỏng = ~104h)
    if train_sim_78:
        p_v11 = os.path.join(args.output_dir, "train_v1_1_ratio_1_3.json")
        combined_11 = train_real_entries + train_sim_78
        with open(p_v11, "w", encoding="utf-8") as f:
            for item in combined_11:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        dur_v11 = sum(e["duration"] for e in combined_11) / 3600.0
        print(f"    [+] v1.1 (Tỷ lệ 1 : 3):     {p_v11} ({len(combined_11)} files - {dur_v11:.1f}h)")

    # v1.2: Ratio 1:5 (26h Thật + 130h Phỏng = ~156h)
    if train_sim_130:
        p_v12 = os.path.join(args.output_dir, "train_v1_2_ratio_1_5.json")
        combined_12 = train_real_entries + train_sim_130
        with open(p_v12, "w", encoding="utf-8") as f:
            for item in combined_12:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        dur_v12 = sum(e["duration"] for e in combined_12) / 3600.0
        print(f"    [+] v1.2 (Tỷ lệ 1 : 5):     {p_v12} ({len(combined_12)} files - {dur_v12:.1f}h)")

    # v1.3: Ratio 1:8 (26h Thật + 208h Phỏng = ~234h)
    if train_sim_208:
        p_v13 = os.path.join(args.output_dir, "train_v1_3_ratio_1_8.json")
        combined_13 = train_real_entries + train_sim_208
        with open(p_v13, "w", encoding="utf-8") as f:
            for item in combined_13:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        dur_v13 = sum(e["duration"] for e in combined_13) / 3600.0
        print(f"    [+] v1.3 (Tỷ lệ 1 : 8):     {p_v13} ({len(combined_13)} files - {dur_v13:.1f}h)")

    # 4. Tùy chọn copy file vật lý nếu người dùng yêu cầu
    if args.copy_files:
        print("\n[*] Đang sao chép file vật lý vào các thư mục tương ứng...")
        splits = [("train", train_pairs), ("dev", dev_pairs), ("test", test_pairs)]
        for split_name, pairs in splits:
            s_audio = os.path.join(args.output_dir, split_name, "audio")
            s_rttm = os.path.join(args.output_dir, split_name, "rttm")
            os.makedirs(s_audio, exist_ok=True)
            os.makedirs(s_rttm, exist_ok=True)
            for w, r in pairs:
                shutil.copy2(w, os.path.join(s_audio, os.path.basename(w)))
                shutil.copy2(r, os.path.join(s_rttm, os.path.basename(r)))
            print(f"    [+] Đã copy {len(pairs)} cặp file vào {args.output_dir}/{split_name}/")

    print("\n" + "=" * 70)
    print("✅ TOÀN BỘ CÔNG TÁC PHÂN CHIA VÀ TẠO MANIFEST ĐÃ HOÀN TẤT THÀNH CÔNG!")
    print(f"📁 Kiểm tra thư mục đầu ra tại: {os.path.abspath(args.output_dir)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
