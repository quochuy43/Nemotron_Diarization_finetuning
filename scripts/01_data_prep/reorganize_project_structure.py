#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script tái cấu trúc dự án Diarization sang chuẩn MLOps / Production-Grade Architecture.
- Di chuyển dữ liệu sang thư mục data/ (tách biệt hoàn toàn khỏi code).
- Di chuyển scripts sang thư mục scripts/01_data_prep/.
- Tự động cập nhật 100% đường dẫn trong tất cả các file Manifest JSON sang cấu trúc mới.

Tác giả: Senior AI Engineer
"""

import os
import glob
import json
import shutil

ROOT_DIR = os.path.abspath(os.path.dirname(__file__)).replace("\\", "/")

def move_safe(src, dst):
    """Di chuyển file/folder an toàn."""
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print(f"[*] Di chuyển: {src} -> {dst}")
        shutil.move(src, dst)
    else:
        print(f"[-] Bỏ qua (không tồn tại): {src}")

def update_manifest_paths(manifest_path, path_replacements):
    """Cập nhật đường dẫn trong 1 file manifest jsonl."""
    if not os.path.exists(manifest_path):
        return 0, 0
    
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))
    
    updated_count = 0
    missing_count = 0

    for item in entries:
        for key in ["audio_filepath", "rttm_filepath"]:
            if item.get(key):
                p = str(item[key]).replace("\\", "/")
                for old_prefix, new_prefix in path_replacements.items():
                    if old_prefix in p:
                        p = p.replace(old_prefix, new_prefix)
                item[key] = p
                
                # Kiểm tra xem file có thực sự tồn tại ở vị trí mới
                if not os.path.exists(p):
                    missing_count += 1
                else:
                    updated_count += 1

    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in entries:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    return len(entries), missing_count

def main():
    print("=" * 70)
    print("🚀 BẮT ĐẦU TÁI CẤU TRÚC DỰ ÁN SANG CHUẨN MLOPS PRODUCTION-GRADE")
    print("=" * 70)

    # 1. Tạo các thư mục kiến trúc mới
    dirs_to_create = [
        "data/real",
        "data/synthetic",
        "data/manifests",
        "scripts/01_data_prep",
        "scripts/02_baseline_eval",
        "scripts/03_training",
        "experiments/baselines",
        "experiments/checkpoints",
    ]
    for d in dirs_to_create:
        os.makedirs(os.path.join(ROOT_DIR, d), exist_ok=True)

    # 2. Di chuyển dữ liệu sang data/
    print("\n--- [BƯỚC 1/3] Di chuyển dữ liệu sang data/ ---")
    move_safe(os.path.join(ROOT_DIR, "real_data"), os.path.join(ROOT_DIR, "data", "real"))
    move_safe(os.path.join(ROOT_DIR, "synthetic_78"), os.path.join(ROOT_DIR, "data", "synthetic", "synthetic_78"))
    move_safe(os.path.join(ROOT_DIR, "synthetic_130"), os.path.join(ROOT_DIR, "data", "synthetic", "synthetic_130"))
    move_safe(os.path.join(ROOT_DIR, "synthetic_208"), os.path.join(ROOT_DIR, "data", "synthetic", "synthetic_208"))
    
    # Di chuyển data_splits sang data/manifests
    if os.path.exists(os.path.join(ROOT_DIR, "data_splits")):
        # Copy các file bên trong data_splits sang data/manifests
        for f in glob.glob(os.path.join(ROOT_DIR, "data_splits", "*")):
            dst = os.path.join(ROOT_DIR, "data", "manifests", os.path.basename(f))
            shutil.move(f, dst)
        os.rmdir(os.path.join(ROOT_DIR, "data_splits"))
        print("[*] Đã chuyển toàn bộ nội dung data_splits -> data/manifests")

    # 3. Di chuyển Scripts sang scripts/01_data_prep/
    print("\n--- [BƯỚC 2/3] Gom Scripts vào scripts/01_data_prep/ ---")
    scripts = [
        "download_and_prepare_japanese_data.py",
        "download_jvs.py",
        "download_travel_agency.py",
        "synthetic_data_generator.py",
        "split_and_prepare_manifests.py",
    ]
    for s in scripts:
        src = os.path.join(ROOT_DIR, s)
        dst = os.path.join(ROOT_DIR, "scripts", "01_data_prep", s)
        move_safe(src, dst)

    # 4. Cập nhật đường dẫn trong toàn bộ các file Manifest JSON
    print("\n--- [BƯỚC 3/3] Chuẩn hóa toàn bộ đường dẫn trong các file Manifest JSON ---")
    
    # Bảng quy đổi đường dẫn cũ -> mới
    replacements = {
        f"{ROOT_DIR}/real_data": f"{ROOT_DIR}/data/real",
        f"{ROOT_DIR}/synthetic_78": f"{ROOT_DIR}/data/synthetic/synthetic_78",
        f"{ROOT_DIR}/synthetic_130": f"{ROOT_DIR}/data/synthetic/synthetic_130",
        f"{ROOT_DIR}/synthetic_208": f"{ROOT_DIR}/data/synthetic/synthetic_208",
    }

    manifest_files = glob.glob(os.path.join(ROOT_DIR, "data", "**", "*.json"), recursive=True)
    total_files_checked = 0
    total_missing = 0

    for mf in manifest_files:
        mf_norm = mf.replace("\\", "/")
        total_entries, missing = update_manifest_paths(mf_norm, replacements)
        total_files_checked += total_entries
        total_missing += missing
        print(f"    [+] {os.path.relpath(mf, ROOT_DIR)}: {total_entries} entries (Lỗi file: {missing})")

    print("\n" + "=" * 70)
    if total_missing == 0:
        print("🎉 TÁI CẤU TRÚC DỰ ÁN THÀNH CÔNG RỰC RỠ!")
        print("✅ 100% ĐƯỜNG DẪN AUDIO VÀ RTTM TỒN TẠI HỢP LỆ VÀ SẠCH SẼ!")
    else:
        print(f"⚠️ Cảnh báo: Có {total_missing} file không tìm thấy, vui lòng kiểm tra lại.")
    print("=" * 70)

if __name__ == "__main__":
    main()
