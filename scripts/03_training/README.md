# Giai Đoạn 4 & 5: Huấn Luyện Grid Search & Checkpoint Averaging (SWA)

Thư mục này chứa toàn bộ cấu hình và mã nguồn phục vụ việc huấn luyện tinh chỉnh (**Fine-tuning**) mô hình `nvidia/Nemotron-3-Diarization` (Ultra-Low Latency 0.32s) cho tiếng Nhật, quét theo dõi ~50 checkpoints trên tập Dev và tạo mô hình xuất xưởng **v2.0 (Final SWA Model)**.

---

## 1. Cấu Trúc File & Vai Trò

| File / Thư mục | Vai Trò & Chức Năng |
| :--- | :--- |
| `configs/sortformer_streaming_032s.yaml` | **Cấu hình huấn luyện YAML**: Định nghĩa siêu tham số mô hình (độ trễ 0.32s, đóng băng 16/31 lớp encoder), bộ tối ưu AdamW ($lr=1e-5$), hàm loss Hungarian PIT-BCE và 4 phiên bản ma trận thử nghiệm (v1.0 đến v1.3). |
| `run_train_grid.py` | **Thực thi huấn luyện Giai đoạn 5**: Quản lý vòng lặp huấn luyện phân tầng, tính toán Hungarian PIT Loss, tự động lưu và đánh giá đều đặn ~50 Checkpoints trên tập Dev (43 files), ghi nhận đường cong hội tụ `convergence_log.json` và lưu danh sách `top_checkpoints.json`. |
| `checkpoint_averaging_swa.py` | **Cộng trung bình trọng số Giai đoạn 6 (v2.0 SWA)**: Lấy Top-5 checkpoints tốt nhất của phiên bản v1.x, tính trung bình trọng số $W_{SWA} = \frac{1}{K}\sum W_k$, kiểm thử độc lập trên tập **Test Set đóng băng 100% (Zero Leakage)** và xuất bảng nghiệm thu Leaderboard. |

---

## 2. Đặc Tả Kỹ Thuật Huấn Luyện (Theo plan.md)

1. **Kiến trúc mô hình**: `Nemotron-3-Diarization` (31-layer Transformer Encoder với RoPE).
2. **Cấu hình Streaming Ultra-Low Latency (0.32s = 320ms)**:
   - `chunk_len = 3` (240 ms)
   - `chunk_right_context = 1` (80 ms lookahead)
   - `spkcache_len = 264` (Arrival-Order Speaker Cache)
   - `fifo_len = 264`
   - `spkcache_update_period = 222`
3. **Chiến lược đóng băng tầng (Layer Freezing)**:
   - Đóng băng nửa dưới (**16 / 31 lớp Encoder đầu tiên**) để bảo tồn đặc trưng biểu diễn âm học tiền huấn luyện.
   - Chỉ cập nhật trọng số của 15 lớp Encoder phía trên và các khối phân loại/speaker cache.
4. **Hàm mất mát hoán vị bất biến (Hungarian PIT Loss)**:
   - Giải bài toán gán nhãn tối ưu giữa 8 kênh dự đoán và 8 kênh nhãn thực tế bằng thuật toán Hungarian Matching trước khi tính Weighted BCE Loss.

---

## 3. Hướng Dẫn Vận Hành CLI

### Bước 1: Kiểm thử nhanh chu trình (Dry-run test)
Kiểm tra toàn bộ pipeline từ nạp audio, forward, backward và đánh giá dev trong 10 giây:
```powershell
python scripts/03_training/run_train_grid.py --dry_run
```

### Bước 2: Huấn luyện chính thức theo các Version trong Ma trận
```powershell
# Chạy phiên bản v1.1 (Tỷ lệ Thật 1 : Mô phỏng 3)
python scripts/03_training/run_train_grid.py --version v1_1 --device cuda

# Hoặc chạy phiên bản v1.2 (Tỷ lệ Thật 1 : Mô phỏng 5)
python scripts/03_training/run_train_grid.py --version v1_2 --device cuda

# Hoặc chạy phiên bản v1.0 (Chỉ dùng dữ liệu mô phỏng Synthetic 78h)
python scripts/03_training/run_train_grid.py --version v1_0 --device cuda
```

### Bước 3: Đóng gói mô hình v2.0 SWA & Nghiệm thu trên Test Set Đóng Băng
Sau khi huấn luyện xong một phiên bản v1.x (ví dụ `v1_1`), chạy script sau để cộng trung bình trọng số Top-5 checkpoint và chấm điểm trên `test_manifest.json`:
```powershell
python scripts/03_training/checkpoint_averaging_swa.py \
    --checkpoints_dir experiments/checkpoints/train_v1_1_ratio_1_3/checkpoints \
    --top_k 5 \
    --device cuda
```

---

## 4. Kiểm Tra 3 Tiêu Chí Nghiệm Thu Sản Phẩm (Production SLA)
Script `checkpoint_averaging_swa.py` sẽ tự động đối chiếu:
1. **Tiêu chí 1**: Test DER (Collar 0s) cải thiện ít nhất $1.5\% - 2.0\%$ so với mốc $v0.2$ ($20.45\%$).
2. **Tiêu chí 2**: Điểm DER ở lát cắt khó 3–7 người không bị suy giảm (giải quyết hiện tượng model chỉ học thoại 2 người).
3. **Tiêu chí 3**: Đạt SLA về độ trễ $p_{99} \le 320\text{ms}$ trên GPU Production.
