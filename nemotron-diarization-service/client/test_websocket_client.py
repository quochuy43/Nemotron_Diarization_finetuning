#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TEST CLIENT: GIẢ LẬP STREAMING AUDIO QUA WEBSOCKET ĐẾN SERVER
===============================================================================
Mục đích:
  - Giả lập Microphone hoặc Call Center gửi từng gói PCM 16-bit qua WebSocket.
  - Đo đạc độ trễ mạng + độ trễ suy luận thực tế (Round-Trip Time & Server Latency).
  - In ra màn hình console theo thời gian thực sự kiện ai đang nói lúc nào.
===============================================================================
"""

import sys
import time
import json
import asyncio
import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
import websockets

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


async def stream_audio_file(
    audio_path: str,
    server_url: str = "ws://localhost:8000/ws/diarize",
    chunk_duration_sec: float = 0.24,
    simulate_realtime: bool = True
):
    print("=" * 80)
    print("🎙️ CLIENT GIẢ LẬP STREAMING ÂM THANH QUA WEBSOCKET")
    print(f"[*] Server URL:       {server_url}")
    print(f"[*] Audio File:       {audio_path}")
    print(f"[*] Chunk Duration:   {chunk_duration_sec}s (240ms)")
    print(f"[*] Mô phỏng Realtime:{simulate_realtime}")
    print("=" * 80)

    # Đọc audio
    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data[:, 0]

    # Resample về 16kHz nếu khác
    if sr != 16000:
        import scipy.signal
        num_samples = int(round(len(data) * 16000 / sr))
        data = scipy.signal.resample(data, num_samples)
        sr = 16000

    chunk_samples = int(round(chunk_duration_sec * sr))
    total_samples = len(data)
    total_chunks = int(np.ceil(total_samples / chunk_samples))

    print(f"[*] Tổng thời lượng: {total_samples / sr:.2f}s | Số lượng chunks: {total_chunks}")
    print("[*] Đang kết nối tới WebSocket Server...")

    latencies = []
    events_received = []

    async with websockets.connect(server_url) as ws:
        # Nhận tin nhắn chào mừng
        welcome = await ws.recv()
        print(f"[+] Đã kết nối: {welcome}\n")
        print("⏱️ KẾT QUẢ PHÂN PHỐI NGƯỜI NÓI THEO THỜI GIAN THỰC:")
        print("-" * 80)

        async def receiver():
            """Lắng nghe kết quả từ server trả về bất đồng bộ."""
            try:
                while True:
                    msg = await ws.recv()
                    event = json.loads(msg)
                    if event.get("type") == "diarization_event":
                        t_s = event["time_start_sec"]
                        t_e = event["time_end_sec"]
                        spks = event["active_speakers"]
                        lat = event["latency_ms"]
                        latencies.append(lat)
                        events_received.append(event)

                        if spks:
                            spk_str = ", ".join([f"{s['speaker_id']} ({s['confidence']*100:.0f}%)" for s in spks])
                            print(f"  🔊 [{t_s:06.2f}s -> {t_e:06.2f}s]  |  Người nói: {spk_str:<30}  |  Server Latency: {lat:5.1f}ms")
                        else:
                            # Im lặng
                            print(f"  🔇 [{t_s:06.2f}s -> {t_e:06.2f}s]  |  (Im lặng)                       |  Server Latency: {lat:5.1f}ms")
            except websockets.exceptions.ConnectionClosed:
                pass

        # Chạy task nhận song song
        recv_task = asyncio.create_task(receiver())

        # Gửi từng chunk PCM 16-bit
        start_time = time.time()
        for i in range(total_chunks):
            s_idx = i * chunk_samples
            e_idx = min(total_samples, s_idx + chunk_samples)
            chunk_float = data[s_idx:e_idx]

            # Chuyển Float32 sang int16 PCM bytes
            chunk_int16 = (np.clip(chunk_float, -1.0, 1.0) * 32767.0).astype(np.int16)
            pcm_bytes = chunk_int16.tobytes()

            t_send = time.time()
            await ws.send(pcm_bytes)

            if simulate_realtime:
                # Ngủ đúng 240ms để mô phỏng nhịp độ nói tự nhiên
                elapsed = time.time() - t_send
                sleep_dur = max(0.0, chunk_duration_sec - elapsed)
                await asyncio.sleep(sleep_dur)

        # Chờ server xử lý nốt các chunk cuối
        await asyncio.sleep(1.0)
        recv_task.cancel()

    # Thống kê tổng hợp
    total_client_time = time.time() - start_time
    print("-" * 80)
    print("📊 THỐNG KÊ PHIÊN STREAMING WEBSOCKET:")
    print(f"  • Tổng thời gian truyền:       {total_client_time:.2f}s")
    print(f"  • Số sự kiện nhận được:        {len(events_received)} chunks")
    if latencies:
        print(f"  • Độ trễ Server trung bình:    {np.mean(latencies):.2f} ms")
        print(f"  • Độ trễ Server p50 (Median):  {np.percentile(latencies, 50):.2f} ms")
        print(f"  • Độ trễ Server p99 (SLA<=320):{np.percentile(latencies, 99):.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test WebSocket Streaming Diarization Client")
    parser.add_argument("--audio", type=str, required=True, help="Đường dẫn file âm thanh WAV test")
    parser.add_argument("--server", type=str, default="ws://localhost:8000/ws/diarize", help="WebSocket server URL")
    parser.add_argument("--fast", action="store_true", help="Gửi tốc độ tối đa không chờ 240ms real-time")
    args = parser.parse_args()

    asyncio.run(stream_audio_file(
        audio_path=args.audio,
        server_url=args.server,
        simulate_realtime=not args.fast
    ))
