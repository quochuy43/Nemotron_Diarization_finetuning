# NVIDIA Nemotron-3 Diarization Service
### Production Real-Time Streaming Service (0.32s SLA) & NIST RTTM Evaluation Hub

---

## 🌟 1. Giới Thiệu
Dịch vụ cung cấp hệ thống phân tách người nói thời gian thực (**Real-Time Speaker Diarization**) xây dựng trên kiến trúc **NVIDIA Nemotron-3 Diarization**, được tinh chỉnh (fine-tuned) và tối ưu hóa đạt độ trễ cực thấp **~26ms** và chỉ số lỗi **DER 11.23%**.

Hệ thống giao tiếp qua **WebSocket 2 chiều** với kiến trúc **Ring Buffer** và thuật toán **Hungarian Matching**, đảm bảo không bị đảo kênh người nói (Zero Permutation Inversion) trong suốt cuộc trò chuyện.

---

## 📁 2. Cấu Trúc Gói Bàn Giao (Delivery Structure)

```text
nemotron-diarization-service/
├── checkpoints/
│   └── model_swa_v1_3_SWA.pt        # File trọng số mô hình SWA xuất xưởng (378MB)
├── server/
│   ├── websocket_server.py         # FastAPI WebSocket Streaming Service
│   └── frontend/                   # UI Dashboard kiểm thử Live Mic / File
│       ├── index.html
│       ├── css/style.css
│       └── js/app.js
├── client/
│   └── test_websocket_client.py    # Code mẫu tích hợp phía Client (Python)
├── docs/
│   ├── API_SPECIFICATION.md        # Đặc tả chi tiết giao thức WebSocket & REST
│   └── BENCHMARK_REPORT.md         # Báo cáo kỹ thuật: DER, RTF, RAM/VRAM, Concurrency
├── Dockerfile                      # File build Docker container hỗ trợ NVIDIA GPU
├── docker-compose.yml              # File triển khai dịch vụ 1 lệnh
├── requirements.txt                # Thư viện runtime cần thiết
└── README.md                       # Tài liệu hướng dẫn này
```

---

## 💻 3. Yêu Cầu Hệ Thống

* **Hệ điều hành**: Linux (Ubuntu 20.04/22.04), Windows 10/11, hoặc Docker Container.
* **GPU**: NVIDIA GPU có hỗ trợ CUDA (Khuyến nghị NVIDIA T4, L4, A10, RTX 3060 trở lên).
* **VRAM**: Tối thiểu 2GB VRAM (Mô hình chỉ chiếm ~400MB VRAM).
* **Python**: Phiên bản 3.10 trở lên.

---

## 🚀 4. Hướng Dẫn Khởi Chạy

### Cách 1: Chạy trực tiếp với Python

1. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Khởi động Server**:
   ```bash
   python server/websocket_server.py --host 0.0.0.0 --port 8000
   ```
   * *Mặc định server sẽ tự động tìm checkpoint tại `checkpoints/model_swa_v1_3_SWA.pt`*.
   * *Nếu muốn truyền đường dẫn checkpoint khác*:
     ```bash
     python server/websocket_server.py --port 8000 --model /duong/dan/model.pt
     ```

3. **Truy cập Giao diện Web UI**:
   Mở trình duyệt: `http://localhost:8000/`

---

### Cách 2: Triển khai với Docker & Docker Compose (Khuyến nghị cho Production)

1. **Khởi chạy bằng Docker Compose**:
   ```bash
   docker compose up -d --build
   ```

2. **Kiểm tra trạng thái container**:
   ```bash
   docker compose logs -f
   ```

3. **Kiểm tra Healthcheck**:
   ```bash
   curl http://localhost:8000/health
   ```

---

## 🧪 5. Kiểm Thử Hệ Thống (Testing & Integration)

### 5.1. Kiểm thử trên Trình Duyệt Web (Web Playground)
* Truy cập `http://localhost:8000/`.
* Chọn tab **Microphone Trực Tiếp** để nói trực tiếp vào mic, xem đèn loa phát sáng theo thời gian thực.
* Hoặc chọn tab **Stream File WAV / MP3**, chọn file âm thanh và file nhãn `.rttm` tương ứng để bấm tính điểm DER chuẩn NIST ngay lập tức.

### 5.2. Kiểm thử bằng Client Code (Python SDK)
Chạy script giả lập gửi luồng âm thanh qua WebSocket:
```bash
python client/test_websocket_client.py --audio duong/dan/file_test.wav --url ws://localhost:8000/ws/diarize
```

---

## 📑 6. Tóm Tắt API & Giao Thức

| Giao thức | Endpoint | Mô tả |
| :--- | :--- | :--- |
| **WebSocket** | `ws://<HOST>:<PORT>/ws/diarize` | Truyền nhận âm thanh PCM16 và nhận JSON sự kiện |
| **REST POST**| `POST /api/evaluate_rttm` | Chấm điểm RTTM dự đoán với Ground Truth RTTM |
| **REST GET** | `GET /health` | Kiểm tra tình trạng hoạt động của Server & GPU |
| **HTTP GET** | `GET /` | Phục vụ trang Web Dashboard |

*Chi tiết đặc tả các trường dữ liệu và cấu trúc gói tin, xem tại [docs/API_SPECIFICATION.md](docs/API_SPECIFICATION.md).*

---

## 🏆 7. Chỉ Số Nghiệm Thu Kỹ Thuật (SLA & Metrics)

* **Độ trễ suy luận (Chunk Latency)**: **~26ms** (Vượt xa cam kết SLA 320ms).
* **Tốc độ xử lý (RTF)**: **0.0015** (Nhanh gấp 647 lần thời gian thực).
* **Độ chính xác (DER Collar 0.25s)**: **11.23%**.
* **Khả năng gánh tải**: 1 GPU NVIDIA có thể phục vụ **80 – 120 luồng streaming song song**.

*Chi tiết báo cáo đo đạc, xem tại [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md).*
