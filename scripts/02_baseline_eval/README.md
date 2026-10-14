# Giai Đoạn 3: Baseline Zero-shot (v0.1) & Tinh Chỉnh Hậu Xử Lý (v0.2)

Thư mục này chứa toàn bộ mã nguồn phục vụ việc đo đạc mốc tham chiếu ban đầu (**Baseline Zero-shot v0.1**) và tinh chỉnh bộ lọc hậu xử lý (**Post-processing Tuning v0.2**) trên tập **Dev Set** (`data/manifests/dev_manifest.json`, 43 files, 6.16 giờ).

---

## 1. Cấu Trúc File & Vai Trò

| File | Vai Trò & Chức Năng |
| :--- | :--- |
| `run_zero_shot_baseline.py` | **Chạy suy luận v0.1**: Nạp mô hình `nvidia/Nemotron-3-Diarization` ở cấu hình streaming Ultra-Low Latency (0.32s), xuất file RTTM dự đoán và ma trận xác suất thô `.npz`. |
| `tune_postprocessing.py` | **Tinh chỉnh v0.2**: Đọc trực tiếp ma trận xác suất `.npz` từ v0.1, quét Grid Search các siêu tham số `threshold`, `min_duration_on`, `min_duration_off` để tối ưu hóa DER trên tập Dev trong vài giây. |
| `evaluate_metrics.py` | **Tính toán chỉ số chuẩn NIST / NOTSOFAR1**: Đo lường DER (Collar 0s, Collar 0.25s), JER, SCA (Speaker Counting Accuracy), MAE và phân tách theo các lát cắt (Overall, 2 người nói thật, 3–7 người nói mô phỏng). |

---

## 2. Đặc Tả Kỹ Thuật Streaming Ultra-Low Latency (0.32s)

Theo tài liệu chính thức từ NVIDIA cho `Nemotron-3-Diarization`:
* **Chunk Length (`chunk_len`)**: 3 frames (240 ms)
* **Lookahead Context (`chunk_right_context`)**: 1 frame (80 ms)
* **Tổng độ trễ bộ đệm (Input Buffer Latency)**: `(3 + 1) * 80 ms = 320 ms = 0.32s`
* **Arrival-Order Speaker Cache (`spkcache_len`)**: 264 frames
* **FIFO Queue Context (`fifo_len`)**: 264 frames
* **Chu kỳ cập nhật Cache (`spkcache_update_period`)**: 222 frames

---

## 3. Hướng Dẫn Thực Thi

### Bước 3.1: Chạy Đo Đạc Baseline Zero-shot (v0.1)
```powershell
python scripts/02_baseline_eval/run_zero_shot_baseline.py --manifest data/manifests/dev_manifest.json --device cuda
```
* **Chế độ kiểm tra nhanh (Debug 2 files)**:
  ```powershell
  python scripts/02_baseline_eval/run_zero_shot_baseline.py --manifest data/manifests/dev_manifest.json --limit 2
  ```
* **Đầu ra tạo thành**:
  - `experiments/baselines/zero_shot_v0_1/rttm/`: 43 file RTTM dự đoán.
  - `experiments/baselines/zero_shot_v0_1/probabilities/`: 43 file `.npz` chứa ma trận xác suất.
  - `experiments/baselines/zero_shot_v0_1/metrics_report.json`: Báo cáo chỉ số toàn diện (DER tổng, DER 2spk, DER 3–7spk, RTF).

---

### Bước 3.2: Tinh Chỉnh Hậu Xử Lý (v0.2)
```powershell
python scripts/02_baseline_eval/tune_postprocessing.py --manifest data/manifests/dev_manifest.json
```
* **Cơ chế hoạt động**:
  - Tận dụng ma trận xác suất `.npz` đã lưu ở bước 3.1, không chạy lại mạng neural.
  - Quét qua 72 cấu hình `(threshold x min_on x min_off)`.
  - Tự động chọn ra bộ tham số có DER (Collar 0s) thấp nhất.
* **Đầu ra tạo thành**:
  - `experiments/baselines/post_proc_v0_2/best_hyperparameters.json`: Lưu các ngưỡng tối ưu nhất.
  - `experiments/baselines/post_proc_v0_2/rttm/`: 43 file RTTM sau khi áp dụng bộ lọc tối ưu.
  - `experiments/baselines/post_proc_v0_2/metrics_report.json`: Báo cáo so sánh đối đầu giữa **v0.1** và **v0.2**.

---

### Bước 3.3 (Tùy chọn): Đánh Giá Độc Lập Bất Kỳ Thư Mục RTTM Nào
```powershell
python scripts/02_baseline_eval/evaluate_metrics.py --manifest data/manifests/dev_manifest.json --hyp_rttm_dir experiments/baselines/zero_shot_v0_1/rttm
```
