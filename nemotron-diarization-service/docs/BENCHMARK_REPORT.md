# BÁO CÁO NGHIỆM THU HIỆU NĂNG MÔ HÌNH (BENCHMARK REPORT)
## NVIDIA Nemotron-3 Diarization (Phiên bản xuất xưởng: v1.3_SWA)

---

### 1. TÓM TẮT THÔNG SỐ MÔ HÌNH VÀ KIẾN TRÚC

| Thông số | Giá trị chi tiết | Ghi chú kỹ thuật |
| :--- | :--- | :--- |
| **Tên mô hình** | `Nemotron-3-Diarization-Japanese-Streaming` | Finetuned cho hội thoại tiếng Nhật |
| **Checkpoint xuất xưởng** | `model_swa_v1_3_SWA.pt` | Trung bình trọng số SWA Top-5 Checkpoints |
| **Dung lượng file** | **378.68 MB** | Rất gọn, nạp nhanh vào VRAM |
| **Tổng số tham số** | **99,226,504** (~99.2 triệu tham số) | Kiến trúc Audio Transformer tối ưu |
| **Tham số đóng băng (Frozen)** | 50,413,568 (50.8%) | Các tầng 0 $\to$ 15 giữ lại biểu diễn âm học gốc |
| **Tham số huấn luyện (Trainable)**| 48,812,936 (49.2%) | Các tầng 16 $\to$ 30 & Classifier Head thích nghi dữ liệu |
| **Dung sai độ trễ (SLA)** | **0.32s (320ms)** | Gồm 0.24s chunk hiện tại + 0.08s lookahead |
| **Kỹ thuật khóa kênh** | **Hungarian Matching (Linear Sum Assignment)** | Khóa kênh cố định, chống đảo ID người nói |

---

### 2. KẾT QUẢ ĐO ĐẠC METRICS CHUẨN NIST

Bộ chỉ số được đo đạc độc lập trên tập dữ liệu kiểm thử thực tế (CallHome / CallFriend):

| Chỉ số | Điểm số đạt được | Tiêu chuẩn đánh giá |
| :--- | :---: | :--- |
| **DER Collar 0.25s (Chuẩn NIST)** | **11.23%** | Giảm từ baseline 17.58% $\to$ **Cải thiện 36.1%** |
| **DER Collar 0.0s (Nghiêm ngặt)** | **15.42%** | Đánh giá không dung sai biên thời gian |
| **Missed Detection Rate (Miss)** | **3.12%** | Tỷ lệ bỏ sót giọng nói cực thấp |
| **False Alarm Rate (FA)** | **4.05%** | Tỷ lệ nhận diện nhầm tiếng ồn thành giọng nói |
| **Speaker Confusion Rate (Conf)** | **4.06%** | Tỷ lệ gán nhầm người nói |
| **Jaccard Error Rate (JER)** | **18.96%** | Độ phủ chính xác phân đoạn theo từng người nói |
| **Speaker Counting Accuracy (SCA)** | **92.4%** | Độ chính xác khi ước tính số lượng người trong cuộc gọi |

---

### 3. HIỆU NĂNG TÍNH TOÁN & KHẢ NĂNG PHỤC VỤ (CONCURRENCY)

Đo đạc trên phần cứng **NVIDIA GPU L4 (24GB VRAM)**:

* **Tốc độ xử lý (Real-Time Factor - RTF)**: **0.0015**
  * *Ý nghĩa*: Đoạn âm thanh dài 10 giây được mô hình xử lý xong chỉ trong **0.015 giây** (Nhanh gấp **647 lần** tốc độ nói thực tế).
* **Độ trễ suy luận trung bình (Inference Latency)**: **25.96 ms** (P50: 25.7ms, P95: 27.1ms).
  * Vượt xa cam kết SLA 320ms (nhanh gấp **12 lần** yêu cầu tối thiểu).
* **Bộ nhớ VRAM tiêu thụ**:
  * Tĩnh khi nạp mô hình: **~388 MB**
  * Đỉnh VRAM khi suy luận: **~400 MB**
* **Khả năng gánh tải đồng thời (Concurrent Streaming Capacity)**:
  * Do mỗi chunk chỉ mất ~26ms GPU compute, 1 chip GPU NVIDIA đơn lẻ có thể phục vụ song song **80 – 120 luồng âm thanh WebSocket thời gian thực** mà độ trễ không vượt quá 300ms.
