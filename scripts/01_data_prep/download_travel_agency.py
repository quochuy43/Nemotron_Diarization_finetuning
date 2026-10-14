import os
import sys
import io
import shutil
import xml.etree.ElementTree as ET
from huggingface_hub import snapshot_download

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi font
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass


def eaf_to_rttm(eaf_path: str, output_rttm_path: str, file_id: str):
    """Phân tích file ELAN (.eaf) và xuất ra file chuẩn NIST RTTM 10 cột."""
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    # Ánh xạ TIME_SLOT_ID sang giây
    time_slots = {}
    for ts in root.findall(".//TIME_SLOT"):
        ts_id = ts.get("TIME_SLOT_ID")
        val_ms = ts.get("TIME_VALUE")
        if ts_id and val_ms is not None:
            time_slots[ts_id] = float(val_ms) / 1000.0

    segments = []
    for tier in root.findall(".//TIER"):
        tier_id = tier.get("TIER_ID", "speaker")
        for ann in tier.findall(".//ALIGNABLE_ANNOTATION"):
            ts1 = ann.get("TIME_SLOT_REF1")
            ts2 = ann.get("TIME_SLOT_REF2")
            if ts1 in time_slots and ts2 in time_slots:
                t_start = time_slots[ts1]
                t_end = time_slots[ts2]
                duration = t_end - t_start
                if duration > 0.01:
                    segments.append((t_start, duration, tier_id))

    # Sắp xếp theo dòng thời gian tăng dần
    segments.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(output_rttm_path), exist_ok=True)
    with open(output_rttm_path, "w", encoding="utf-8") as f:
        for t_start, duration, spk in segments:
            f.write(
                f"SPEAKER {file_id} 1 {t_start:.3f} {duration:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            )


def download_and_prepare():
    repo_id = "IngCrowd/japanese-roleplay-travel-agency"
    base_dir = os.path.abspath("ingcrowd_japanese_roleplay_travel_agency")
    audio_dir = os.path.join(base_dir, "audio")
    rttm_dir = os.path.join(base_dir, "rttm")
    raw_dir = os.path.join(base_dir, "_raw")

    print("=" * 70)
    print(f"[*] BẮT ĐẦU TẢI DATASET: {repo_id} (~725 MB)")
    print("=" * 70)

    # 1. Tải toàn bộ repo về thư mục tạm _raw
    print("[1/3] Đang tải từ Hugging Face CDN...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=raw_dir,
    )
    print("[+] Tải thành công về thư mục tạm.")

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)

    # 2. Xử lý chuẩn hóa và dọn dẹp
    print("[2/3] Đang xử lý tạo RTTM và chuẩn hóa tên file audio...")
    dialogue_count = 0

    for root, _, files in os.walk(raw_dir):
        for file in files:
            # Xử lý file annotation .eaf -> .rttm
            if file.endswith(".eaf"):
                eaf_path = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                out_rttm = os.path.join(rttm_dir, f"{file_id}.rttm")
                eaf_to_rttm(eaf_path, out_rttm, file_id)

            # Xử lý file mixed audio: bỏ hậu tố '_audio.wav' -> '.wav'
            elif file.endswith("_audio.wav"):
                src_wav = os.path.join(root, file)
                file_id = file.replace("_audio.wav", "")
                dst_wav = os.path.join(audio_dir, f"{file_id}.wav")
                shutil.copy2(src_wav, dst_wav)
                dialogue_count += 1
                print(f"    -> Đã chuẩn hóa: {file_id}.wav & {file_id}.rttm")

    # 3. Dọn dẹp thư mục tạm _raw (xóa các file individual_audio và eaf thô)
    print("[3/3] Đang dọn dẹp các file rác...")
    try:
        shutil.rmtree(raw_dir)
    except Exception as e:
        print(f"Lưu ý: Không thể xóa thư mục tạm: {e}")

    print("\n" + "=" * 70)
    print("🎉 HOÀN TẤT CHUẨN BỊ DATASET ROLEPLAY TRAVEL AGENCY!")
    print(f"    - Tổng số cuộc đối thoại: {dialogue_count} files")
    print(f"    - Thư mục Audio sạch: {audio_dir}")
    print(f"    - Thư mục Ground Truth RTTM: {rttm_dir}")
    print("=" * 70)


if __name__ == "__main__":
    download_and_prepare()
