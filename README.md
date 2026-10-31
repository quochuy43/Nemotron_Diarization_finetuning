# Real-time Japanese Speaker Diarization (Nemotron-3 0.32s)

> **Mục tiêu dự án**: Tối ưu hóa mô hình **Streaming Speaker Diarization** cho tiếng Nhật đạt độ trễ cực thấp **0.32s (Ultra-Low Latency)** dựa trên kiến trúc `nvidia/Nemotron-3-Diarization`, kết hợp **35.61h dữ liệu thật** (4 datasets) và **416h dữ liệu mô phỏng đa người nói** (JVS Corpus 100 speakers).

---

## 1. Tổng Quan Kiến Trúc & Đặc Tả Kỹ Thuật

* **Mô hình cốt lõi**: `nvidia/Nemotron-3-Diarization` (~100M tham số, 31-layer Transformer encoder với RoPE, hỗ trợ tối đa 8 người nói).
* **Cấu hình Streaming Ultra-Low Latency (0.32s = 320 ms)**:
  * `chunk_len = 3` (240 ms)
  * `chunk_right_context = 1` (80 ms)
  * **Độ trễ bộ đệm (Input Buffer Latency)**: `(chunk_len + chunk_right_context) * 80 ms = 4 * 80 ms = 320 ms = 0.32s`
  * `spkcache_len = 264` (Arrival-Order Speaker Cache - duy trì nhận diện người nói theo thứ tự xuất hiện)
  * `fifo_len = 264` (Ngữ cảnh khung thời gian gần nhất)
  * `spkcache_update_period = 222` (Chu kỳ cập nhật cache)
* **Thước đo đánh giá chuẩn quốc tế (NIST / DIHARD / NOTSOFAR1)**:
  * **DER (Diarization Error Rate)**: Báo cáo `Collar 0.0s` (chính) và `Collar 0.25s` (kèm phân rã Miss, False Alarm, Speaker Confusion).
  * **JER (Jaccard Error Rate)**, **SCA (Speaker Counting Accuracy %)**, **MAE (Speaker Counting Error)**.
  * **RTF (Real-Time Factor)**: Tốc độ xử lý / thời lượng audio.

> [!TIP]
> 📖 **Tài liệu & Báo cáo Kỹ thuật Chuyên sâu**:
> - 📊 [**Báo Cáo Kỹ Thuật Tổng Hợp (Technical Report)**](docs/TECHNICAL_REPORT.md): So sánh đối đầu 4 phiên bản ma trận thực nghiệm (v1.0 -> v1.3), phân tích âm học và quy luật scaling dữ liệu.
> - 🚀 [**Báo Cáo Nghiệm Thu Xuất Xưởng v1.3 SWA (Production Benchmark)**](docs/PRODUCTION_BENCHMARK.md): Đo đạc độ trễ phần cứng $p_{99} = 29.58\text{ms}$ trên NVIDIA L4 GPU và chuẩn SLA 0.32s.

---

## 2. Cấu Trúc Thư Mục Chuẩn MLOps & Research Layout

```text
Nemotron_Diarization_finetuning/
│
├── configs/                                 <-- Cấu hình huấn luyện
│   └── sortformer_streaming_032s.yaml       <-- Config NeMo 0.32s streaming
│
├── data/                                    <-- TẦNG DỮ LIỆU & MANIFESTS
│   ├── raw/                                 <-- Kho âm thanh gốc (jvs_corpus 100 speakers)
│   ├── real/                                <-- 4 bộ dữ liệu thật (CallHome, CallFriend, IngCrowd - 35.61h)
│   ├── synthetic/                           <-- Dữ liệu mô phỏng sạch sẽ từ JVS Corpus (416h)
│   └── manifests/                           <-- Manifests JSON Lines chuẩn NeMo (POSIX forward slashes)
│       ├── dev_manifest.json                <-- Đề thi thử (43 files - 6.16h)
│       ├── test_manifest.json               <-- Kỳ thi thật đóng băng 100% (43 files - 5.44h)
│       ├── train_real_manifest.json         <-- 125 files Thật (~26.05h)
│       ├── train_v1_0_sim_only.json         <-- 1,520 files Mô phỏng (~77.8h)
│       ├── train_v1_1_ratio_1_3.json        <-- 1,645 files (103.9h | Thật 1 : Phỏng 3)
│       ├── train_v1_2_ratio_1_5.json        <-- 2,685 files (157.2h | Thật 1 : Phỏng 5)
│       └── train_v1_3_ratio_1_8.json        <-- 4,245 files (237.0h | Thật 1 : Phỏng 8)
│
├── docs/                                    <-- TẦNG BÁO CÁO NGHIÊN CỨU & KỸ THUẬT
│   ├── TECHNICAL_REPORT.md                  <-- Báo cáo đối đầu 4 phiên bản & 4 phát hiện âm học
│   ├── PRODUCTION_BENCHMARK.md              <-- Báo cáo nghiệm thu mô hình xuất xưởng v1.3 SWA
│   ├── research_plan.md                     <-- Bản thiết kế kiến trúc ban đầu
│   └── experiments_history/                 <-- Báo cáo chi tiết từng mốc (v1.0 -> v1.3) & JSON metrics
│
├── scripts/                                 <-- MÃ NGUỒN PIPELINE THEO CÁC GIAI ĐOẠN
│   ├── 01_data_prep/                        <-- [TIỀN XỬ LÝ & TẠO DỮ LIỆU DSP]
│   │   ├── download_jvs.py                  <-- Tải JVS Corpus 100 speakers
│   │   ├── download_and_prepare_japanese_data.py <-- Tải & chuẩn hóa ELAN sang RTTM
│   │   ├── split_and_prepare_manifests.py   <-- Phân chia Stratified Hold-out Split (Zero Data Leakage)
│   │   └── synthetic_data_generator.py      <-- Engine DSP sinh hội thoại mô phỏng (overlap 12-15%)
│   │
│   ├── 02_baseline_eval/                    <-- [BASELINE & TINH CHỈNH HẬU XỬ LÝ]
│   │   ├── run_zero_shot_baseline.py        <-- Chạy suy luận streaming 0.32s, trích xuất .npz
│   │   ├── tune_postprocessing.py           <-- Quét Grid Search 72 siêu tham số
│   │   └── evaluate_metrics.py              <-- Đánh giá NIST DER Collar 0s/0.25s, JER, SCA
│   │
│   ├── 03_training/                         <-- [HUẤN LUYỆN GRID SEARCH & SWA]
│   │   ├── configs/sortformer_streaming_032s.yaml
│   │   ├── run_train_grid.py                <-- Quét checkpoint đều đặn (~50 CPs / run)
│   │   └── checkpoint_averaging_swa.py      <-- Checkpoint Averaging (SWA) Top-5 CPs
│   │
│   └── 04_deployment/                       <-- [STREAMING INFERENCE & BENCHMARKING]
│       ├── evaluate_streaming.py            <-- Hardware Profiler trên GPU NVIDIA L4
│       ├── stream_diarization.py            <-- CLI mô phỏng stream audio từng chunk
│       ├── websocket_server.py              <-- FastAPI WebSocket Streaming Server
│       └── test_websocket_client.py         <-- Client kiểm thử chịu tải streaming
│
├── .gitignore
├── LICENSE                                  <-- MIT License
├── README.md                                <-- Tài liệu tổng quan (File này)
└── requirements.txt                         <-- Ràng buộc thư viện tái lập môi trường
```

---

## 3. Chiến Lược Dữ Liệu & Chống Rò Rỉ (Zero Data Leakage)

1. **Khắc phục bài toán thiếu dữ liệu đông người tiếng Nhật**:
   * Dữ liệu thực tế tiếng Nhật (`CallHome`, `CallFriend`, `IngCrowd`) 100% chỉ có hội thoại **2 người**.
   * Dùng `synthetic_data_generator.py` ghép từ 100 giọng JVS bản xứ để bù đắp các case **3–7 người nói**.
   * DSP tối ưu: Gọt sạch khoảng lặng phòng thu (RMS Energy threshold -32 dBFS, margin 25 ms), nhịp ngắt nghỉ tiếng Nhật (Latching 0.02s–0.12s), tỷ lệ overlap vàng 12%–15%, và Peak Normalization -1 dBFS chống rè tiếng.
2. **Nguyên tắc phân tầng Zero Data Leakage**:
   * Cả tập **Dev** (43 files) và **Test** (43 files) đều gồm: **23 file Thật (2 người)** + **20 file Mô phỏng (3–7 người)**.
   * 40 session mô phỏng này đã được **khóa và rút hoàn toàn khỏi toàn bộ các tập Train**.
   * Tập **Test Set được đóng băng 100%**, chỉ dùng nghiệm thu cuối cùng.

---

## 4. Môi Trường Hoạt Động & Ràng Buộc Thư Viện (System Requirements)

* **Python**: 3.12 (Khuyến nghị môi trường Windows/Linux 64-bit).
* **Phần cứng**: Hỗ trợ NVIDIA CUDA GPU (Đã kiểm thử mượt mà trên **RTX 3050 Laptop GPU 4GB VRAM**).
* **Ràng buộc phụ thuộc then chốt (Crucial Dependency Constraints)**:
  * `torch >= 2.6.0` (Đang dùng bản chuẩn CUDA: `torch-2.6.0+cu124`).
  * `numpy < 2.5` (Bắt buộc dùng `numpy==2.4.6`. Nếu lên `>= 2.5` sẽ làm gãy `numba 0.64.0` vốn là dependency của `librosa` trích xuất Mel-spectrogram).
  * `transformers >= 5.19.0` (Hỗ trợ native cho kiến trúc `Nemotron3DiarizationAudioModel` và `AutoProcessor` với RoPE attention).
  * `pyannote.metrics` & `pyannote.core` (Dùng để tính DER, JER, NIST RTTM scoring).

---

## 5. Hiện Trạng Triển Khai & Kết Quả Đạt Được

### Tiến độ các giai đoạn:
* [x] **Giai đoạn 1**: Thu thập 35.61h âm thanh thật + Sinh 416h âm thanh mô phỏng sạch sẽ (**HOÀN TẤT 100%**).
* [x] **Giai đoạn 2**: Phân chia Stratified Hold-out Split & tạo 6 file NeMo manifests JSON Lines (**HOÀN TẤT 100%**).
* [x] **Giai đoạn 3**: Viết bộ công cụ & kiểm thử thành công Baseline v0.1 và Tinh chỉnh Hậu xử lý v0.2 (**HOÀN TẤT 100%**).
* [x] **Giai đoạn 4**: Chuẩn bị file cấu hình huấn luyện `sortformer_streaming_032s.yaml` (**HOÀN TẤT 100%**).
* [x] **Giai đoạn 5**: Huấn luyện Grid Search theo các tỷ lệ (1:3, 1:5, 1:8) và script quét 50 Checkpoints trên Dev (`run_train_grid.py`) (**HOÀN TẤT 100%**).
* [x] **Giai đoạn 6**: Checkpoint Averaging (SWA v2.0) và nghiệm thu chốt trên Test Set (`checkpoint_averaging_swa.py`) (**HOÀN TẤT 100%**).

### Kết quả Benchmark Chính thức trên Toàn bộ Tập Dev Set (43 files):
* **Baseline Zero-shot (v0.1)**: `DER Collar 0s = 20.89%` (Miss: 3.18% | FA: 6.42% | Conf: 11.29%) | 2 người: `18.77%` | 3–7 người: `29.61%`.
* **Post-processing Tuning (v0.2)**: `DER Collar 0s = 20.45%` (giảm -0.44%) | 2 người: `18.40%` | 3–7 người: `28.85%`.
* **Bộ siêu tham số hậu xử lý tối ưu chuẩn (v0.2 & SWA)**:
  * `threshold = 0.45` (Ngưỡng quyết định)
  * `min_duration_on = 0.05s` (Thời lượng nói tối thiểu)
  * `min_duration_off = 0.15s` (Khoảng lặng nối liền)

> [!NOTE]
> *Ghi chú tốc độ suy luận (RTF)*: Kiểm thử trên GPU cá nhân RTX 3050 Laptop đạt RTF = **`0.0038`** (~260x real-time speed, xử lý 20 phút âm thanh chỉ trong 4.64 giây).

---

## 6. Hướng Dẫn Vận Hành Dành Cho AI Agents & Developers

### 1. Chạy đo đạc Baseline Zero-shot v0.1:
```powershell
# Chạy toàn bộ 43 files Dev set
python scripts/02_baseline_eval/run_zero_shot_baseline.py --manifest data/manifests/dev_manifest.json

# Chạy test nhanh 2 files debug
python scripts/02_baseline_eval/run_zero_shot_baseline.py --manifest data/manifests/dev_manifest.json --limit 2
```

### 2. Chạy quét Grid Search tối ưu hậu xử lý v0.2:
```powershell
# Quét 72 tổ hợp tham số từ ma trận xác suất .npz đã lưu ở v0.1 (chạy trong ~30s không cần chạy lại model)
python scripts/02_baseline_eval/tune_postprocessing.py --manifest data/manifests/dev_manifest.json
```

### 3. Đánh giá độc lập bất kỳ thư mục RTTM nào:
```powershell
python scripts/02_baseline_eval/evaluate_metrics.py --manifest data/manifests/dev_manifest.json --hyp_rttm_dir experiments/baselines/zero_shot_v0_1/rttm
```

---

## 7. Quy Tắc Bắt Buộc Dành Cho AI Coding Agents (Agent Guidelines)

1. **Định dạng đường dẫn trong Manifest**: Luôn sử dụng dấu gạch xuôi chuẩn POSIX `/` (ví dụ `data/real/callhome_jpn/...`), **tuyệt đối không dùng dấu gạch ngược Windows `\\`** để đảm bảo tương thích chéo giữa Windows và Linux GPU Cluster/Docker.
2. **Định dạng Manifest JSON Lines**: Mỗi dòng là một JSON object hoàn chỉnh riêng biệt. Không bọc toàn bộ file trong mảng `[...]` và không dùng multi-line indent khi ghi file manifest NeMo.
3. **Bảo toàn tính năng Zero Leakage**: Khi tạo thêm bất kỳ dữ liệu huấn luyện mới nào, không bao giờ được chạm vào hoặc thêm các file đã nằm trong `data/manifests/dev_manifest.json` và `data/manifests/test_manifest.json`.
4. **Không chạy lệnh nặng ngầm**: Khi code các tính năng mới, luôn hỗ trợ tham số `--limit` hoặc chế độ dry-run để test logic trước khi chạy trọn vẹn dữ liệu lớn.
