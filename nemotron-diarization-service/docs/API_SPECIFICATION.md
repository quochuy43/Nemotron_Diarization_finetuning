# ĐẶC TẢ KỸ THUẬT GIAO DIỆN LẬP TRÌNH (API SPECIFICATION)
## Dịch Vụ Phân Tách Người Nói Thời Gian Thực - Nemotron-3 Diarization

---

### 1. TỔNG QUAN HỆ THỐNG
Dịch vụ cung cấp khả năng phát hiện và phân đoạn người nói theo thời gian thực (Real-Time Speaker Diarization) đạt chuẩn SLA **0.32s** với độ trễ xử lý GPU trung bình chỉ **~26ms**. 

* **Kiến trúc**: FastAPI ASGI + WebSocket Full-Duplex + Native Speaker Cache (`ultra_low_latency`: chunk 3 frames + right context 1 frame, spkcache/FIFO 264 frames).
* **Giao thức âm thanh**: PCM 16-bit, Sampling Rate 16,000 Hz, Kênh Mono.
* **Chu kỳ gửi gói đề xuất**: 240ms (tương đương 3,840 mẫu = 7,680 bytes).
* **Số người nói hỗ trợ tối đa**: 8 người nói đồng thời / phiên.

---

### 2. ENDPOINT CHÍNH: WEBSOCKET STREAMING

#### `ws://<HOST>:<PORT>/ws/diarize`
Kênh truyền nhận 2 chiều giữa Audio Client (Trình duyệt, Call Center, PBX, App) và Diarization Server.

#### Query Parameters:
| Tham số | Kiểu dữ liệu | Mặc định | Ý nghĩa |
| :--- | :--- | :--- | :--- |
| `client_id` | String | Auto UUID | Định danh phiên làm việc (Session ID) |
| `threshold` | Float | `0.45` | Ngưỡng xác suất kích hoạt người nói |

---

### 3. GIAO THỨC TRUYỀN NHẬN QUA WEBSOCKET

#### 3.1. Phía Client gửi lên Server
Client có thể gửi 2 loại Frame:

##### A. Binary Frame (Dữ liệu âm thanh)
* Định dạng: Mảng bytes thô (Raw bytes) chứa âm thanh **PCM 16-bit Little-Endian, 16kHz, Mono**.
* Dung lượng chuẩn mỗi gói: 240ms = 3,840 samples = **7,680 bytes**.
* *Lưu ý*: Server tự động tích lũy audio, nên client có thể gửi gói nhỏ hơn (ví dụ 10ms - 100ms) nếu cần.

##### B. Text Frame (Gói điều khiển JSON)
* **Khởi tạo lại phiên làm việc (Reset buffer về 0.0s)**:
  ```json
  {
    "action": "reset"
  }
  ```
  *Tác dụng*: Xóa sạch audio đệm, **speaker cache** và lịch sử xác suất, đưa mốc thời gian chunk tiếp theo bắt đầu chính xác từ `0.00s`.
* **Kiểm tra kết nối (Heartbeat / Ping)**:
  ```json
  {
    "action": "ping"
  }
  ```
* **Kết thúc phiên và lấy RTTM đầy đủ**:
  ```json
  {
    "action": "finalize",
    "uri": "ten_file_hoac_phien"
  }
  ```
  *Tác dụng*: Server xử nốt phần audio dư (chunk cuối, không có look-ahead), rồi dựng RTTM của **cả phiên** từ ma trận xác suất frame 10ms qua đúng bộ lọc hậu xử lý đã tối ưu lúc train (`threshold=0.45`, `min_duration_on=0.05s`, `min_duration_off=0.15s`), và trả về message `final_rttm`.

  Đây là **đầu ra chuẩn** để chấm DER. Việc gộp các `diarization_event` 240ms ở phía client chỉ đạt độ phân giải 240ms và không áp dụng `min_duration_on/off`, nên kém chính xác hơn.

  Bộ lọc này chỉ nhìn lại tối đa 150ms nên vẫn chạy được online với độ trễ đầu ra 150ms — nó **không** dùng thông tin tương lai xa.

##### Tham số query khi mở WebSocket
| Tham số | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `threshold` | `0.45` | Ngưỡng kích hoạt tiếng nói |
| `min_duration_on` | `0.05` | Bỏ đoạn thoại ngắn hơn 50ms |
| `min_duration_off` | `0.15` | Nối khoảng lặng ngắn hơn 150ms |
| `emit_frame_probs` | `false` | Kèm ma trận xác suất frame 10ms (24×8) trong mỗi event |
| `client_id` | tự sinh | Định danh phiên |
| `key` | — | API key nếu server bật xác thực |

---

#### 3.2. Phía Server phản hồi về Client

##### A. Thông điệp Chào Mừng (Ngay khi kết nối thành công):
```json
{
  "type": "connection_established",
  "session_id": "sess_8f29ab10",
  "sample_rate": 16000,
  "chunk_duration_sec": 0.24,
  "sla_latency_sec": 0.32,
  "threshold": 0.45,
  "message": "Sẵn sàng nhận dữ liệu âm thanh PCM 16-bit 16kHz mono qua WebSocket."
}
```

##### B. Sự Kiện Diarization (Diarization Event - Trả về sau mỗi chunk 240ms):
```json
{
  "type": "diarization_event",
  "session_id": "sess_8f29ab10",
  "chunk_index": 142,
  "time_start_sec": 34.08,
  "time_end_sec": 34.32,
  "active_speakers": [
    {
      "speaker_id": "SPEAKER_00",
      "channel_index": 0,
      "confidence": 0.9852
    }
  ],
  "num_active_speakers": 1,
  "unique_speakers_seen": ["SPEAKER_00", "SPEAKER_01"],
  "latency_ms": 25.84,
  "sla_passed": true
}
```
*Giải thích các trường*:
* `time_start_sec`, `time_end_sec`: Mốc thời gian bắt đầu và kết thúc của chunk âm thanh hiện tại.
* `active_speakers`: Danh sách những người đang nói trong khoảng thời gian này kèm độ tin cậy (`confidence` từ 0.0 đến 1.0).
* `latency_ms`: Thời gian suy luận GPU thực tế trên chunk này (thường ~24–28ms).
* `sla_passed`: `true` nếu `latency_ms <= 320ms` (cam kết không gây trễ luồng).

##### C. Xác nhận Reset Thành Công (Reset ACK):
```json
{
  "type": "reset_ack",
  "session_id": "sess_8f29ab10"
}
```

---

### 4. REST API ENDPOINTS

#### 4.1. Chấm điểm NIST RTTM với Ground Truth
`POST /api/evaluate_rttm`

* **Headers**: `Content-Type: application/json`
* **Request Body**:
  ```json
  {
    "reference_rttm": "SPEAKER audio1 1 0.80 3.23 <NA> <NA> SPK_01 <NA> <NA>\nSPEAKER audio1 1 4.76 2.62 <NA> <NA> SPK_02 <NA> <NA>",
    "hypothesis_rttm": "SPEAKER audio1 1 0.72 3.30 <NA> <NA> SPEAKER_00 <NA> <NA>\nSPEAKER audio1 1 4.80 2.50 <NA> <NA> SPEAKER_01 <NA> <NA>"
  }
  ```
* **Response (200 OK)**:
  ```json
  {
    "status": "success",
    "metrics": {
      "der_collar0": 15.42,
      "der_collar025": 11.23,
      "miss_rate_c0": 3.12,
      "false_alarm_c0": 4.05,
      "confusion_c0": 8.25,
      "jer": 18.96,
      "ref_speakers": 2,
      "hyp_speakers": 2,
      "speaker_count_match": true,
      "speaker_count_diff": 0,
      "ref_speech_duration_sec": 42.5
    }
  }
  ```

#### 4.2. Health Check
`GET /health`

* **Response (200 OK)**:
  ```json
  {
    "status": "UP",
    "model_loaded": true,
    "device": "cuda",
    "active_sessions": 3,
    "sla_target_ms": 320.0
  }
  ```

#### 4.3. Demo Web UI
`GET /`
* Phục vụ trang Dashboard trực quan tại `http://<HOST>:<PORT>/`.
