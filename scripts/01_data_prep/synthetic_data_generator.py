import os
import sys
import io
import random
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Đảm bảo in tiếng Việt trên console Windows
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass


class JVSDatasetIndexer:
    """Quét và lập chỉ mục toàn bộ file âm thanh của 100 người trong JVS Corpus."""
    
    def __init__(self, jvs_root: str):
        self.jvs_root = jvs_root
        self.speakers: Dict[str, Dict[str, List[str]]] = {}
        self.speaker_gender: Dict[str, str] = {}
        self._index_dataset()

    def _index_dataset(self):
        # 1. Đọc giới tính từ gender_f0range.txt nếu có
        gender_file = os.path.join(self.jvs_root, "gender_f0range.txt")
        if os.path.exists(gender_file):
            with open(gender_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0].startswith("jvs"):
                        self.speaker_gender[parts[0]] = parts[1]

        # 2. Quét các file âm thanh của từng người (ưu tiên 24kHz16bit)
        speaker_dirs = sorted([
            d for d in os.listdir(self.jvs_root) 
            if os.path.isdir(os.path.join(self.jvs_root, d)) and d.startswith("jvs")
        ])

        print(f"[*] Đang lập chỉ mục dữ liệu cho {len(speaker_dirs)} speakers trong JVS Corpus...")

        for spk in speaker_dirs:
            spk_dir = os.path.join(self.jvs_root, spk)
            self.speakers[spk] = {
                "parallel": glob.glob(os.path.join(spk_dir, "parallel100", "wav24kHz16bit", "*.wav")),
                "nonpara": glob.glob(os.path.join(spk_dir, "nonpara30", "wav24kHz16bit", "*.wav")),
                "whisper": glob.glob(os.path.join(spk_dir, "whisper10", "wav24kHz16bit", "*.wav")),
                "falset": glob.glob(os.path.join(spk_dir, "falset10", "wav24kHz16bit", "*.wav")),
            }

        total_audio_files = sum(
            len(files) 
            for spk_data in self.speakers.values() 
            for files in spk_data.values()
        )
        print(f"[+] Lập chỉ mục thành công: {len(self.speakers)} speakers | {total_audio_files:,} câu thoại sạch.")


class ConversationSimulator:
    """Engine mô phỏng hội thoại đa người với mốc thời gian và độ chồng lấn tự nhiên."""

    def __init__(self, indexer: JVSDatasetIndexer, target_sr: int = 16000, native_sr: int = 24000):
        self.indexer = indexer
        self.target_sr = target_sr
        self.native_sr = native_sr
        self.all_spk_ids = list(indexer.speakers.keys())

    def _sample_num_speakers(self) -> int:
        """
        Phân phối số người theo chiến lược kế hoạch:
        - 2 người: ~40%
        - 3–4 người: ~35%
        - 5–7 người: ~25%
        """
        r = random.random()
        if r < 0.40:
            return 2
        elif r < 0.75:
            return random.choice([3, 4])
        else:
            return random.choice([5, 6, 7])

    def _trim_silence(self, y: np.ndarray, top_db: float = 32.0, frame_len: int = 512, hop_len: int = 128) -> np.ndarray:
        """
        Cắt bỏ khoảng lặng studio đầu và đuôi (JVS corpus có ~0.5s silence cố định ở cả đầu và đuôi).
        Giữ lại đệm an toàn ~25ms để bảo toàn âm gió, bật hơi và đuôi nguyên âm.
        """
        if len(y) <= frame_len:
            return y
        num_frames = (len(y) - frame_len) // hop_len + 1
        if num_frames <= 0:
            return y
        
        frames = np.lib.stride_tricks.as_strided(
            y,
            shape=(num_frames, frame_len),
            strides=(y.strides[0] * hop_len, y.strides[0])
        )
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-9)
        max_rms = np.max(rms)
        if max_rms < 1e-5:
            return y
        
        rms_db = 20 * np.log10(rms / max_rms)
        active_frames = np.where(rms_db > -top_db)[0]
        if len(active_frames) == 0:
            return y
        
        # Đệm an toàn ~25ms (400 samples tại 16kHz) ở cả 2 đầu
        margin_samples = int(0.025 * self.target_sr)
        start_sample = max(0, active_frames[0] * hop_len - margin_samples)
        end_sample = min(len(y), (active_frames[-1] + 1) * hop_len + margin_samples)
        return y[start_sample:end_sample]

    def _load_and_resample(self, wav_path: str) -> np.ndarray:
        """Đọc audio 24kHz, chuyển về 16kHz và cắt sạch khoảng lặng studio ở 2 đầu."""
        data, sr = sf.read(wav_path, dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]  # Mono
        if sr != self.target_sr:
            # Resample từ native_sr (24k) về target_sr (16k) tỷ lệ 2/3
            data = resample_poly(data, self.target_sr // 1000, sr // 1000)
        
        # Cắt gọt khoảng lặng chết thừa ở đầu và đuôi file gốc
        data = self._trim_silence(data)
        return data

    def generate_session(
        self, 
        session_id: str, 
        target_duration_sec: float = 180.0
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Tạo 1 cuộc hội thoại hoàn chỉnh kéo dài target_duration_sec giây.
        Trả về:
            - mixed_audio: Mảng numpy 1D tần số 16kHz
            - rttm_lines: Danh sách các dòng định dạng RTTM 10 cột
        """
        num_speakers = self._sample_num_speakers()
        selected_speakers = random.sample(self.all_spk_ids, num_speakers)

        # Gán mức Gain chênh lệch âm lượng cố định cho mỗi người trong session (±3 dB đến ±6 dB)
        speaker_gains = {
            spk: 10 ** (random.uniform(-4.0, 4.0) / 20.0) 
            for spk in selected_speakers
        }

        # Bộ đệm âm thanh tổng (dư ra 15s để tránh tràn index)
        total_samples = int((target_duration_sec + 15.0) * self.target_sr)
        mixed_audio = np.zeros(total_samples, dtype=np.float32)
        
        segments = []  # (start_sec, dur_sec, speaker_id)
        current_time = random.uniform(0.2, 0.6)  # Bắt đầu vào việc nhanh sau 0.2 - 0.6s
        last_speaker = None

        while current_time < target_duration_sec:
            # 1. Chọn người nói tiếp theo (85% đổi người, 15% người cũ nói tiếp)
            if last_speaker is None or random.random() < 0.85:
                available = [s for s in selected_speakers if s != last_speaker]
                speaker = random.choice(available)
            else:
                speaker = last_speaker

            # 2. Chọn ngẫu nhiên loại câu (80% bình thường, 10% whisper, 10% falset)
            cat_choice = random.choices(
                ["parallel", "nonpara", "whisper", "falset"],
                weights=[0.55, 0.35, 0.05, 0.05]
            )[0]
            
            pool = self.indexer.speakers[speaker][cat_choice]
            if not pool:
                pool = self.indexer.speakers[speaker]["parallel"]
            
            wav_file = random.choice(pool)
            audio_clip = self._load_and_resample(wav_file)
            
            # Áp dụng gain âm lượng của người đó
            audio_clip = audio_clip * speaker_gains[speaker]
            clip_len_sec = len(audio_clip) / self.target_sr

            # 3. Đặt vào trục thời gian
            start_sample = int(current_time * self.target_sr)
            end_sample = start_sample + len(audio_clip)

            if end_sample <= total_samples:
                mixed_audio[start_sample:end_sample] += audio_clip
                segments.append((current_time, clip_len_sec, speaker))

            # 4. Quyết định lượt tiếp theo (Gap vs Overlap) - Tối ưu hóa phản xạ tự nhiên tiếng Nhật
            is_overlap = random.random() < 0.25 and current_time + clip_len_sec < target_duration_sec
            if is_overlap and speaker != last_speaker:
                # Đổi người và nói chồng:
                # - 70% đệm từ / Aizuchi ngắn vào đuôi câu (0.15s - 0.40s)
                # - 30% cướp lời / phản hồi dồn dập (0.30s - 0.70s)
                if random.random() < 0.70:
                    overlap_dur = random.uniform(0.15, min(0.40, clip_len_sec * 0.35))
                else:
                    overlap_dur = random.uniform(0.30, min(0.70, clip_len_sec * 0.50))

                current_time += max(0.1, clip_len_sec - overlap_dur)
            else:
                # Ngắt nghỉ (Gap): Vì audio gốc đã được gọt sạch khoảng lặng chết ~0.5s ở 2 đầu,
                # khoảng cách pause_dur giờ đây chính là khoảng lặng thực tế tai nghe thấy!
                if speaker == last_speaker:
                    # Cùng 1 người nói tiếp câu 2: lấy hơi tự nhiên (0.15s - 0.40s)
                    pause_dur = random.uniform(0.15, 0.40)
                else:
                    # Đổi người đối đáp theo nhịp giao tiếp tiếng Nhật:
                    r_gap = random.random()
                    if r_gap < 0.45:
                        # 45% Nối lượt tức thì / phản hồi cực nhanh (Latching: 0.02s - 0.12s)
                        pause_dur = random.uniform(0.02, 0.12)
                    elif r_gap < 0.90:
                        # 45% Nhịp đối thoại thông thường (0.12s - 0.30s)
                        pause_dur = random.uniform(0.12, 0.30)
                    else:
                        # 10% Suy nghĩ / ngập ngừng ngắn (0.30s - 0.55s)
                        pause_dur = random.uniform(0.30, 0.55)

                current_time += clip_len_sec + pause_dur

            last_speaker = speaker

        # Cắt đúng kích thước thực tế
        actual_length_sec = segments[-1][0] + segments[-1][1] + 1.0 if segments else target_duration_sec
        actual_samples = int(actual_length_sec * self.target_sr)
        mixed_audio = mixed_audio[:actual_samples]

        # Chuẩn hóa âm lượng chống vỡ tiếng (Peak Normalization to -1 dB)
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 0.01:
            target_peak = 10 ** (-1.0 / 20.0)  # -1 dBFS
            mixed_audio = (mixed_audio / max_val) * target_peak

        # Sắp xếp các đoạn RTTM theo thời gian bắt đầu
        segments.sort(key=lambda x: x[0])
        rttm_lines = [
            f"SPEAKER {session_id} 1 {t_start:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"
            for t_start, dur, spk in segments
        ]

        return mixed_audio, rttm_lines


def generate_synthetic_dataset(
    jvs_root: str,
    output_dir: str,
    total_hours: float,
    session_duration_min: float = 3.0
):
    """Tạo toàn bộ dataset mô phỏng với số giờ yêu cầu."""
    os.makedirs(output_dir, exist_ok=True)
    audio_dir = os.path.join(output_dir, "audio")
    rttm_dir = os.path.join(output_dir, "rttm")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(rttm_dir, exist_ok=True)

    indexer = JVSDatasetIndexer(jvs_root)
    simulator = ConversationSimulator(indexer)

    session_sec = session_duration_min * 60.0
    total_sessions = int((total_hours * 3600.0) / session_sec)
    total_sessions = max(1, total_sessions)

    print("=" * 70)
    print(f"🚀 BẮT ĐẦU TẠO DỮ LIỆU MÔ PHỎNG:")
    print(f"    - Mục tiêu: {total_hours:.1f} giờ ({total_sessions} cuộc hội thoại)")
    print(f"    - Độ dài mỗi cuộc: ~{session_duration_min:.1f} phút ({session_sec:.0f} giây)")
    print(f"    - Thư mục xuất: {output_dir}")
    print("=" * 70)

    manifest_entries = []

    for i in range(1, total_sessions + 1):
        session_id = f"sim_jvs_{i:05d}"
        wav_path = os.path.join(audio_dir, f"{session_id}.wav")
        rttm_path = os.path.join(rttm_dir, f"{session_id}.rttm")

        # Tạo âm thanh và RTTM
        audio_data, rttm_lines = simulator.generate_session(session_id, session_sec)

        # Lưu audio 16kHz
        sf.write(wav_path, audio_data, 16000, subtype="PCM_16")

        # Lưu RTTM
        with open(rttm_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rttm_lines) + "\n")

        # Tính số lượng speaker trong file
        spks_in_session = len(set(line.split()[7] for line in rttm_lines))
        actual_dur = len(audio_data) / 16000.0

        manifest_entries.append({
            "audio_filepath": os.path.abspath(wav_path),
            "offset": 0.0,
            "duration": round(actual_dur, 2),
            "label": "infer",
            "text": "-",
            "num_speakers": spks_in_session,
            "rttm_filepath": os.path.abspath(rttm_path),
            "uem_filepath": None
        })

        if i % 10 == 0 or i == total_sessions:
            elapsed_hours = (i * session_sec) / 3600.0
            print(f"    -> Đã tạo: {i}/{total_sessions} sessions (~{elapsed_hours:.2f}h) [Sample: {spks_in_session} spks, {actual_dur:.1f}s]...")

    # Ghi file manifest.json cho NeMo
    manifest_path = os.path.join(output_dir, "manifest.json")
    import json
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print(f"🎉 HOÀN TẤT TẠO DỮ LIỆU MÔ PHỎNG:")
    print(f"    - Tổng số file tạo ra: {total_sessions} sessions")
    print(f"    - Tổng thời lượng xấp xỉ: {total_hours:.1f} giờ")
    print(f"    - File manifest NeMo: {manifest_path}")
    print(f"    - Thư mục Audio: {audio_dir}")
    print(f"    - Thư mục RTTM: {rttm_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo hội thoại mô phỏng từ JVS Corpus cho Speaker Diarization")
    parser.add_argument(
        "--jvs_dir", 
        type=str, 
        default="jvs_corpus/jvs_ver1", 
        help="Đường dẫn đến thư mục jvs_ver1"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="synthetic_dataset", 
        help="Thư mục xuất dữ liệu mô phỏng"
    )
    parser.add_argument(
        "--hours", 
        type=float, 
        default=1.0, 
        help="Số giờ âm thanh muốn tạo (ví dụ: 1.0 để test, hoặc 78.0 / 130.0 / 200.0)"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        default=3.0, 
        help="Thời lượng trung bình mỗi cuộc hội thoại tính bằng phút (mặc định: 3.0 phút)"
    )

    args = parser.parse_args()
    generate_synthetic_dataset(
        jvs_root=args.jvs_dir,
        output_dir=args.output_dir,
        total_hours=args.hours,
        session_duration_min=args.duration
    )
