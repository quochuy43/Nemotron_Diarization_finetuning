#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script: build_test_manifests.py
Mục đích:
  1. Tạo data/manifests/test1_manifest.json từ tập test 43 files hiện tại.
  2. Quét toàn bộ thư mục data/javis/ (60 files: 9_files_v4, youtube_3 -> youtube_7)
     và tạo data/manifests/test2_manifest.json chuẩn NVIDIA NeMo.
  3. Tạo test1_files.txt và test2_files.txt để dễ tra cứu.
===============================================================================
"""

import os
import sys
import json
import shutil
from pathlib import Path
import soundfile as sf

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thư mục gốc repo
repo_root = Path(__file__).resolve().parent.parent.parent
manifests_dir = repo_root / "data" / "manifests"
manifests_dir.mkdir(parents=True, exist_ok=True)

test_manifest_src = manifests_dir / "test_manifest.json"
test1_manifest_dest = manifests_dir / "test1_manifest.json"
test1_files_dest = manifests_dir / "test1_files.txt"

test2_manifest_dest = manifests_dir / "test2_manifest.json"
test2_files_dest = manifests_dir / "test2_files.txt"

javis_dir = repo_root / "data" / "javis"


def build_test1():
    """Tạo test1_manifest.json từ test_manifest.json (43 files)."""
    if not test_manifest_src.exists():
        print(f"[!] Cảnh báo: Không tìm thấy {test_manifest_src}")
        return

    # Copy sang test1_manifest.json
    shutil.copyfile(test_manifest_src, test1_manifest_dest)
    print(f"[+] Đã tạo: {test1_manifest_dest}")

    # Tạo test1_files.txt
    lines = []
    with open(test_manifest_src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                lines.append(f"{Path(item['audio_filepath']).name} ({item.get('duration', 0):.1f}s, {item.get('num_speakers', 2)} spks)")

    with open(test1_files_dest, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[+] Đã tạo danh sách: {test1_files_dest} ({len(lines)} files)")


def build_test2():
    """Quét toàn bộ data/javis/ và tạo test2_manifest.json."""
    if not javis_dir.exists():
        print(f"[!] Cảnh báo: Không tìm thấy thư mục {javis_dir}")
        return

    items = []
    file_list = []

    sub_dirs = sorted([d for d in javis_dir.iterdir() if d.is_dir()])
    print(f"[*] Đang quét {len(sub_dirs)} thư mục trong {javis_dir.name}...")

    for sub in sub_dirs:
        audio_dir = sub / "audio"
        rttm_dir = sub / "rttm"
        if not audio_dir.exists():
            continue

        wav_files = sorted(audio_dir.glob("*.wav"))
        print(f"  • Thư mục {sub.name}: tìm thấy {len(wav_files)} file wav...")

        for wav_path in wav_files:
            stem = wav_path.stem
            rttm_path = rttm_dir / f"{stem}.rttm" if rttm_dir.exists() else None

            # Đọc thời lượng âm thanh
            try:
                info = sf.info(str(wav_path))
                duration = round(info.duration, 2)
            except Exception as e:
                print(f"    [!] Lỗi đọc file {wav_path.name}: {e}")
                continue

            # Đọc số lượng người nói từ RTTM
            speakers = set()
            if rttm_path and rttm_path.exists():
                with open(rttm_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 8 and parts[0] == "SPEAKER":
                            speakers.add(parts[7])
            
            num_spks = len(speakers) if speakers else 2

            # Chuẩn hóa đường dẫn tương đối chuẩn cho manifest
            rel_audio = f"data/javis/{sub.name}/audio/{wav_path.name}"
            rel_rttm = f"data/javis/{sub.name}/rttm/{rttm_path.name}" if rttm_path and rttm_path.exists() else None

            manifest_entry = {
                "audio_filepath": rel_audio,
                "offset": 0.0,
                "duration": duration,
                "label": "infer",
                "text": "-",
                "num_speakers": num_spks,
                "rttm_filepath": rel_rttm,
                "uem_filepath": None,
                "subset": sub.name
            }
            items.append(manifest_entry)
            file_list.append(f"{sub.name}/{wav_path.name} ({duration:.1f}s, {num_spks} spks)")

    # Ghi ra file test2_manifest.json
    with open(test2_manifest_dest, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n[+] ĐÃ TẠO THÀNH CÔNG: {test2_manifest_dest} ({len(items)} files, {sum(x['duration'] for x in items)/3600:.2f} giờ)")

    # Ghi ra file test2_files.txt
    with open(test2_files_dest, "w", encoding="utf-8") as f:
        f.write("\n".join(file_list) + "\n")
    print(f"[+] Đã tạo danh sách: {test2_files_dest}")


if __name__ == "__main__":
    print("=" * 80)
    print("🚀 KHỞI TẠO TEST1 VÀ TEST2 MANIFESTS")
    print("=" * 80)
    build_test1()
    print("-" * 80)
    build_test2()
    print("=" * 80)
