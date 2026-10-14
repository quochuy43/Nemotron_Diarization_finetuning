import os
import sys
import io
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
from huggingface_hub import snapshot_download, hf_hub_download

# Fix Windows console UTF-8 output
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass


def eaf_to_rttm(eaf_path: str, output_rttm_path: str, file_id: str):
    """Chuyển đổi file ELAN (.eaf) sang file chuẩn RTTM 10 cột."""
    tree = ET.parse(eaf_path)
    root = tree.getroot()

    # Thu thập mốc thời gian (TIME_SLOT_ID -> giây)
    time_slots = {}
    for ts in root.findall(".//TIME_SLOT"):
        time_slot_id = ts.get("TIME_SLOT_ID")
        time_val_ms = ts.get("TIME_VALUE")
        if time_slot_id and time_val_ms is not None:
            time_slots[time_slot_id] = float(time_val_ms) / 1000.0

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

    # Sắp xếp theo thời gian bắt đầu
    segments.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(output_rttm_path), exist_ok=True)
    with open(output_rttm_path, "w", encoding="utf-8") as f:
        for t_start, duration, spk in segments:
            f.write(
                f"SPEAKER {file_id} 1 {t_start:.3f} {duration:.3f} <NA> <NA> {spk} <NA> <NA>\n"
            )


def download_ingcrowd():
    print("=" * 70)
    print(">>> 1. ĐANG TẢI DATASET: IngCrowd/japanese-themed-dialogue (~665 MB)")
    print("=" * 70)

    target_dir = os.path.abspath("ingcrowd_japanese_themed_dialogue")
    os.makedirs(target_dir, exist_ok=True)

    local_dir = snapshot_download(
        repo_id="IngCrowd/japanese-themed-dialogue",
        repo_type="dataset",
        local_dir=target_dir,
    )
    print(f"[+] Tải hoàn tất IngCrowd về: {local_dir}")

    # Chuyển đổi các file .eaf sang .rttm
    rttm_dir = os.path.join(target_dir, "rttm")
    os.makedirs(rttm_dir, exist_ok=True)

    converted_count = 0
    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".eaf"):
                eaf_path = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                out_rttm = os.path.join(rttm_dir, f"{file_id}.rttm")
                eaf_to_rttm(eaf_path, out_rttm, file_id)
                converted_count += 1
                print(f"    -> Đã tạo RTTM cho: {file_id}.rttm")

    print(f"[HOÀN TẤT INGCROWD] Đã xử lý {converted_count} file đối thoại và tạo RTTM tại: {rttm_dir}\n")


def download_callfriend():
    print("=" * 70)
    print(">>> 2. ĐANG TẢI DATASET: TalkBank CallFriend Japanese (~1.35 GB)")
    print("=" * 70)

    out_base = os.path.abspath("callfriend_jpn")
    audio_dir = os.path.join(out_base, "audio")
    rttm_dir = os.path.join(out_base, "rttm")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)

    parquet_files = [
        "jpn/data-00000-of-00003.parquet",
        "jpn/data-00001-of-00003.parquet",
        "jpn/data-00002-of-00003.parquet",
    ]

    total_extracted = 0

    for idx, p_file in enumerate(parquet_files, 1):
        print(f"[*] Đang tải và xử lý phần {idx}/3 ({p_file})...")
        local_parquet = hf_hub_download(
            repo_id="talkbank/callfriend",
            filename=p_file,
            repo_type="dataset",
        )
        df = pd.read_parquet(local_parquet)

        for _, row in df.iterrows():
            audio_dict = row["audio"]
            raw_path = audio_dict.get("path", "")
            
            # Đặt tên file chuẩn
            if raw_path:
                file_id = Path(raw_path).stem
            else:
                file_id = f"callfriend_jpn_{total_extracted:04d}"

            # Lưu file audio WAV
            wav_path = os.path.join(audio_dir, f"{file_id}.wav")
            with open(wav_path, "wb") as f_wav:
                f_wav.write(audio_dict["bytes"])

            # Tạo file RTTM tương ứng
            starts = row["timestamps_start"]
            ends = row["timestamps_end"]
            speakers = row["speakers"]

            segments = []
            for t_start, t_end, spk in zip(starts, ends, speakers):
                dur = float(t_end) - float(t_start)
                if dur > 0.01:
                    segments.append((float(t_start), dur, str(spk)))

            # Sắp xếp theo thời gian
            segments.sort(key=lambda x: x[0])

            rttm_path = os.path.join(rttm_dir, f"{file_id}.rttm")
            with open(rttm_path, "w", encoding="utf-8") as f_rttm:
                for t_start, dur, spk in segments:
                    f_rttm.write(
                        f"SPEAKER {file_id} 1 {t_start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>\n"
                    )

            total_extracted += 1
            if total_extracted % 5 == 0:
                print(f"    -> Đã trích xuất: {total_extracted} cuộc gọi (Audio + RTTM)...")

    print(f"\n[HOÀN TẤT CALLFRIEND] Đã trích xuất thành công {total_extracted} cuộc gọi:")
    print(f"    - Audio WAV: {audio_dir}")
    print(f"    - Ground truth RTTM: {rttm_dir}\n")


def main():
    try:
        download_ingcrowd()
        download_callfriend()
        print("=" * 70)
        print("🎉 TẤT CẢ DATASET TIẾNG NHẬT ĐÃ ĐƯỢC TẢI VÀ CHUYỂN ĐỔI RTTM XONG!")
        print("=" * 70)
    except Exception as e:
        print(f"\n[LỖI]: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
