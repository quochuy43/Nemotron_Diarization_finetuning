#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Evaluation: stream N audio files qua WebSocket, sinh predicted RTTM,
so sánh với ground truth và tính metrics trung bình.

Usage:
    python client/batch_evaluate.py \
        --audio-dir 9_files_v4/audio \
        --rttm-dir  9_files_v4/rttm \
        --server ws://localhost:8000/ws/diarize \
        --fast
"""

import sys
import os
import time
import json
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import soundfile as sf

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import websockets
except ImportError:
    sys.exit("websockets package required: pip install websockets")

try:
    import httpx
except ImportError:
    httpx = None


# ---------------------------------------------------------------------------
# RTTM builder: chuyển diarization_event stream → RTTM text
# ---------------------------------------------------------------------------

def events_to_rttm(events: List[Dict[str, Any]], file_id: str) -> str:
    """
    Fallback thô: gom các diarization_event 240ms thành đoạn RTTM liên tục.

    CHỈ dùng khi server không trả về `final_rttm`. Hàm này quyết định ở độ phân
    giải 240ms và không áp dụng min_duration_on / min_duration_off, nên kém chính
    xác hơn RTTM do server dựng từ xác suất frame 10ms.
    """
    # Tạo danh sách (start, end, speaker_id) từ mỗi event
    raw_segments: List[tuple] = []
    for ev in events:
        t_s = ev["time_start_sec"]
        t_e = ev["time_end_sec"]
        for spk in ev.get("active_speakers", []):
            raw_segments.append((t_s, t_e, spk["speaker_id"]))

    if not raw_segments:
        return ""

    # Nhóm theo speaker, sort theo thời gian
    from collections import defaultdict
    by_speaker: Dict[str, List[tuple]] = defaultdict(list)
    for s, e, spk in raw_segments:
        by_speaker[spk].append((s, e))

    rttm_lines = []
    for spk, segs in sorted(by_speaker.items()):
        segs.sort()
        merged = [segs[0]]
        for s, e in segs[1:]:
            prev_s, prev_e = merged[-1]
            if s <= prev_e + 0.02:  # merge nếu gap <= 20ms
                merged[-1] = (prev_s, max(prev_e, e))
            else:
                merged.append((s, e))
        for s, e in merged:
            dur = round(e - s, 3)
            if dur > 0.001:
                rttm_lines.append(
                    f"SPEAKER {file_id} 1 {s:.3f} {dur:.3f} <NA> <NA> {spk} <NA> <NA>"
                )

    # Sort theo start time
    rttm_lines.sort(key=lambda l: float(l.split()[3]))
    return "\n".join(rttm_lines) + "\n"


# ---------------------------------------------------------------------------
# Metrics: gọi server REST API hoặc tính local qua pyannote
# ---------------------------------------------------------------------------

async def compute_metrics_via_api(
    base_url: str, ref_rttm: str, hyp_rttm: str
) -> Optional[Dict[str, Any]]:
    """Gọi /api/evaluate_rttm trên server."""
    if httpx is None:
        return None
    url = f"{base_url}/api/evaluate_rttm"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json={
            "reference_rttm": ref_rttm,
            "hypothesis_rttm": hyp_rttm,
        })
        data = resp.json()
        if data.get("status") == "success":
            return data["metrics"]
        raise RuntimeError(data.get("message", "Unknown API error"))


def compute_metrics_local(ref_rttm: str, hyp_rttm: str) -> Dict[str, Any]:
    """Fallback: tính metrics bằng pyannote local."""
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    def _parse(text, uri):
        ann = Annotation(uri=uri)
        for line in text.strip().splitlines():
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                s, d = float(parts[3]), float(parts[4])
                if d > 0.001:
                    ann[Segment(s, s + d)] = parts[7]
        return ann

    uri = "eval"
    ref = _parse(ref_rttm, uri)
    hyp = _parse(hyp_rttm, uri)

    der_c0 = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    der_c025 = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)

    d0 = der_c0(ref, hyp) * 100
    d025 = der_c025(ref, hyp) * 100
    j = jer(ref, hyp) * 100

    acc = der_c0.accumulated_
    tot = max(acc.get("total", 0), 1e-6)
    miss = acc.get("missed detection", 0) / tot * 100
    fa = acc.get("false alarm", 0) / tot * 100
    conf = acc.get("confusion", 0) / tot * 100

    return {
        "der_collar0": round(d0, 2),
        "der_collar025": round(d025, 2),
        "miss_rate_c0": round(miss, 2),
        "false_alarm_c0": round(fa, 2),
        "confusion_c0": round(conf, 2),
        "jer": round(j, 2),
        "ref_speakers": len(ref.labels()),
        "hyp_speakers": len(hyp.labels()),
        "speaker_count_diff": abs(len(ref.labels()) - len(hyp.labels())),
        "ref_speech_duration_sec": round(tot, 2),
    }


# ---------------------------------------------------------------------------
# Single-file streaming
# ---------------------------------------------------------------------------

async def stream_one_file(
    audio_path: Path,
    server_url: str,
    chunk_duration_sec: float = 0.24,
    simulate_realtime: bool = True,
    api_key: Optional[str] = None,
    file_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stream 1 file qua WebSocket, thu thập diarization_event và RTTM cuối phiên.

    Sau khi gửi hết audio, client gửi {"action": "finalize"} để server xử nốt
    phần dư và dựng RTTM của cả phiên từ xác suất frame 10ms (đúng bộ lọc
    threshold/min_on/min_off của pipeline đánh giá lúc train).
    """
    data, sr = sf.read(str(audio_path), dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        import soxr
        data = soxr.resample(data, sr, 16000, quality='HQ')
        sr = 16000

    chunk_samples = int(round(chunk_duration_sec * sr))
    total_samples = len(data)
    total_chunks = int(np.ceil(total_samples / chunk_samples))
    audio_duration = total_samples / sr

    url = server_url
    if api_key:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}key={api_key}"

    events: List[Dict[str, Any]] = []
    latencies: List[float] = []
    final_msg: Dict[str, Any] = {}
    done = asyncio.Event()

    async with websockets.connect(url, max_size=16 * 1024 * 1024) as ws:
        welcome = await ws.recv()

        async def receiver():
            try:
                while True:
                    msg = await ws.recv()
                    ev = json.loads(msg)
                    if ev.get("type") == "diarization_event":
                        events.append(ev)
                        latencies.append(ev["latency_ms"])
                    elif ev.get("type") == "final_rttm":
                        final_msg.update(ev)
                        done.set()
                    elif ev.get("type") == "error":
                        print(f"  [!] Server error: {ev.get('message')}")
                        done.set()
            except websockets.exceptions.ConnectionClosed:
                done.set()

        recv_task = asyncio.create_task(receiver())

        t0 = time.time()
        for i in range(total_chunks):
            s_idx = i * chunk_samples
            e_idx = min(total_samples, s_idx + chunk_samples)
            # Match web UI PCM16 conversion: s < 0 ? s * 0x8000 : s * 0x7FFF
            chunk_f32 = np.clip(data[s_idx:e_idx], -1.0, 1.0)
            chunk_int16 = np.where(chunk_f32 < 0, chunk_f32 * 32768, chunk_f32 * 32767).astype(np.int16)
            await ws.send(chunk_int16.tobytes())

            if simulate_realtime:
                target = t0 + (i + 1) * chunk_duration_sec
                sleep_dur = target - time.time()
                if sleep_dur > 0:
                    await asyncio.sleep(sleep_dur)

        # Yêu cầu server xử nốt phần dư và trả RTTM cả phiên
        await ws.send(json.dumps({"action": "finalize", "uri": file_id or audio_path.stem}))
        try:
            await asyncio.wait_for(done.wait(), timeout=max(60.0, audio_duration))
        except asyncio.TimeoutError:
            print("  [!] Timeout chờ final_rttm từ server")
        recv_task.cancel()

    wall_time = time.time() - t0

    return {
        "events": events,
        "latencies": latencies,
        "final_rttm": final_msg.get("rttm", ""),
        "final_duration_sec": final_msg.get("duration_sec"),
        "audio_duration_sec": round(audio_duration, 2),
        "wall_time_sec": round(wall_time, 2),
        "total_chunks_sent": total_chunks,
        "total_events_recv": len(events),
    }


# ---------------------------------------------------------------------------
# Main batch runner
# ---------------------------------------------------------------------------

async def run_batch(args):
    audio_dir = Path(args.audio_dir)
    rttm_dir = Path(args.rttm_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(audio_dir.glob("*.wav"))
    if not audio_files:
        sys.exit(f"Không tìm thấy file .wav nào trong {audio_dir}")

    # Xác định base HTTP URL từ WebSocket URL (cho REST API)
    http_base = args.server.replace("ws://", "http://").replace("wss://", "https://")
    http_base = http_base.split("/ws/")[0]

    print("=" * 90)
    print("  BATCH EVALUATION — Nemotron Diarization Service")
    print(f"  Audio dir:  {audio_dir}  ({len(audio_files)} files)")
    print(f"  RTTM dir:   {rttm_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Server:     {args.server}")
    print(f"  Mode:       {'FAST (max speed)' if args.fast else 'Real-time simulation'}")
    print("=" * 90)

    all_metrics: List[Dict[str, Any]] = []
    all_latencies: List[float] = []
    summary_rows: List[Dict[str, Any]] = []

    # Bộ tích lũy GLOBAL (trọng số theo thời lượng + UEM) — đúng cách mà
    # scripts/02_baseline_eval/evaluate_metrics.py dùng khi chấm lúc train.
    # Trung bình cộng DER từng file (cách cũ ở dưới) KHÔNG so sánh được với
    # các con số trong report.md, nên ở đây báo cáo song song cả hai.
    from pyannote.core import Annotation, Segment, Timeline
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    global_der_c0 = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    global_der_c025 = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    global_jer = JaccardErrorRate(collar=0.0, skip_overlap=False)

    def _annot(text: str, uri: str) -> Annotation:
        ann = Annotation(uri=uri)
        for line in text.strip().splitlines():
            parts = line.split()
            if len(parts) >= 8 and parts[0] == "SPEAKER":
                s, d = float(parts[3]), float(parts[4])
                if d > 0.001:
                    ann[Segment(s, s + d)] = parts[7]
        return ann

    for idx, audio_path in enumerate(audio_files, 1):
        file_id = audio_path.stem
        rttm_path = rttm_dir / f"{file_id}.rttm"

        print(f"\n{'─' * 90}")
        print(f"  [{idx}/{len(audio_files)}] {audio_path.name}")
        print(f"  GT RTTM: {'OK' if rttm_path.exists() else 'MISSING'}")

        # 1. Stream audio
        t_start = time.time()
        result = await stream_one_file(
            audio_path=audio_path,
            server_url=args.server,
            simulate_realtime=not args.fast,
            api_key=args.key,
            file_id=file_id,
        )
        stream_elapsed = time.time() - t_start

        events = result["events"]
        latencies = result["latencies"]
        all_latencies.extend(latencies)

        print(f"  Audio: {result['audio_duration_sec']}s | "
              f"Chunks sent: {result['total_chunks_sent']} | "
              f"Events recv: {result['total_events_recv']} | "
              f"Wall time: {stream_elapsed:.1f}s")

        if latencies:
            print(f"  Latency — avg: {np.mean(latencies):.1f}ms | "
                  f"p50: {np.percentile(latencies, 50):.1f}ms | "
                  f"p99: {np.percentile(latencies, 99):.1f}ms")

        # 2. Lấy RTTM do server dựng từ xác suất frame 10ms (ưu tiên),
        #    chỉ fallback về gộp event 240ms nếu server không trả về.
        hyp_rttm = result.get("final_rttm") or ""
        rttm_source = "server final_rttm (frame 10ms + min_on/min_off)"
        if not hyp_rttm.strip():
            hyp_rttm = events_to_rttm(events, file_id)
            rttm_source = "fallback: gộp event 240ms"
        pred_path = output_dir / f"{file_id}.pred.rttm"
        pred_path.write_text(hyp_rttm, encoding="utf-8")
        print(f"  Predicted RTTM [{rttm_source}] saved: {pred_path}")

        # 3. Compute metrics
        if not rttm_path.exists():
            print("  [!] Ground truth RTTM missing — skipping metrics")
            continue

        ref_rttm = rttm_path.read_text(encoding="utf-8")

        if not hyp_rttm.strip():
            print("  [!] Predicted RTTM is empty — skipping metrics")
            continue

        try:
            metrics = await compute_metrics_via_api(http_base, ref_rttm, hyp_rttm)
            if metrics is None:
                metrics = compute_metrics_local(ref_rttm, hyp_rttm)
        except Exception as e:
            print(f"  [!] Metrics computation failed: {e}")
            try:
                metrics = compute_metrics_local(ref_rttm, hyp_rttm)
            except Exception as e2:
                print(f"  [!] Local fallback also failed: {e2}")
                continue

        all_metrics.append(metrics)
        summary_rows.append({"file": file_id, **metrics})

        # Tích lũy GLOBAL với UEM = [0, audio_duration] của chính file này
        _ref, _hyp = _annot(ref_rttm, file_id), _annot(hyp_rttm, file_id)
        _uem = Timeline([Segment(0.0, result["audio_duration_sec"])])
        global_der_c0(_ref, _hyp, uem=_uem)
        global_der_c025(_ref, _hyp, uem=_uem)
        global_jer(_ref, _hyp, uem=_uem)

        print(f"  DER(c=0): {metrics['der_collar0']:.2f}%  |  "
              f"DER(c=0.25): {metrics['der_collar025']:.2f}%  |  "
              f"JER: {metrics['jer']:.2f}%")
        print(f"  Miss: {metrics['miss_rate_c0']:.2f}%  |  "
              f"FA: {metrics['false_alarm_c0']:.2f}%  |  "
              f"Conf: {metrics['confusion_c0']:.2f}%")
        print(f"  Speakers — ref: {metrics['ref_speakers']}  "
              f"hyp: {metrics['hyp_speakers']}  "
              f"diff: {metrics['speaker_count_diff']}")

    # ---------------------------------------------------------------------------
    # Aggregate summary
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("  AGGREGATE RESULTS")
    print(f"{'=' * 90}")

    if not all_metrics:
        print("  No metrics were computed.")
        return

    metric_keys = [
        "der_collar0", "der_collar025", "miss_rate_c0",
        "false_alarm_c0", "confusion_c0", "jer",
    ]
    avg = {}
    for k in metric_keys:
        vals = [m[k] for m in all_metrics if k in m]
        avg[k] = round(np.mean(vals), 2) if vals else 0.0

    n = len(all_metrics)
    speaker_match = sum(1 for m in all_metrics if m.get("speaker_count_diff", 1) == 0)

    def _components(metric) -> Dict[str, float]:
        acc = metric.accumulated_
        tot = max(acc.get("total", 0.0), 1e-6)
        return {
            "der": round(abs(metric) * 100, 2),
            "miss": round(acc.get("missed detection", 0.0) / tot * 100, 2),
            "fa": round(acc.get("false alarm", 0.0) / tot * 100, 2),
            "conf": round(acc.get("confusion", 0.0) / tot * 100, 2),
            "total_speech_sec": round(tot, 2),
        }

    g_c0 = _components(global_der_c0)
    g_c025 = _components(global_der_c025)
    g_jer = round(abs(global_jer) * 100, 2)

    print(f"\n  Files evaluated:      {n}/{len(audio_files)}")
    print(f"  Speaker count match:  {speaker_match}/{n}")

    print("\n  [A] GLOBAL — trọng số theo thời lượng + UEM")
    print("      (cùng cách chấm với scripts/02_baseline_eval/evaluate_metrics.py,")
    print("       đây là con số so sánh được trực tiếp với report.md)")
    print(f"      {'DER (collar=0)':<22} {g_c0['der']:>9.2f}%   "
          f"[miss {g_c0['miss']:.2f} / fa {g_c0['fa']:.2f} / conf {g_c0['conf']:.2f}]")
    print(f"      {'DER (collar=0.25)':<22} {g_c025['der']:>9.2f}%   "
          f"[miss {g_c025['miss']:.2f} / fa {g_c025['fa']:.2f} / conf {g_c025['conf']:.2f}]")
    print(f"      {'JER':<22} {g_jer:>9.2f}%")
    print(f"      {'Tổng thời lượng thoại':<22} {g_c0['total_speech_sec']:>9.2f}s")

    print("\n  [B] TRUNG BÌNH CỘNG TỪNG FILE (không trọng số, không UEM)")
    print("      (cách tổng hợp cũ của script này — giữ lại để đối chiếu các lần chạy trước;")
    print("       KHÔNG so sánh trực tiếp được với report.md)")
    print(f"      {'DER (collar=0)':<22} {avg['der_collar0']:>9.2f}%")
    print(f"      {'DER (collar=0.25)':<22} {avg['der_collar025']:>9.2f}%")
    print(f"      {'Miss Rate':<22} {avg['miss_rate_c0']:>9.2f}%")
    print(f"      {'False Alarm':<22} {avg['false_alarm_c0']:>9.2f}%")
    print(f"      {'Confusion':<22} {avg['confusion_c0']:>9.2f}%")
    print(f"      {'JER':<22} {avg['jer']:>9.2f}%")

    if all_latencies:
        print(f"\n  Latency (all chunks):")
        print(f"    avg: {np.mean(all_latencies):.1f}ms  |  "
              f"p50: {np.percentile(all_latencies, 50):.1f}ms  |  "
              f"p95: {np.percentile(all_latencies, 95):.1f}ms  |  "
              f"p99: {np.percentile(all_latencies, 99):.1f}ms")

    # Per-file table
    print(f"\n  {'─' * 90}")
    print(f"  {'File':<44} {'DER(c0)':>8} {'DER(.25)':>9} {'Miss':>7} {'FA':>7} {'Conf':>7} {'JER':>7} {'Spk':>5}")
    print(f"  {'─' * 90}")
    for row in summary_rows:
        fname = row["file"][:42]
        print(f"  {fname:<44} {row['der_collar0']:>7.2f}% {row['der_collar025']:>8.2f}% "
              f"{row['miss_rate_c0']:>6.2f}% {row['false_alarm_c0']:>6.2f}% "
              f"{row['confusion_c0']:>6.2f}% {row['jer']:>6.2f}% "
              f"{row['ref_speakers']}→{row['hyp_speakers']}")
    print(f"  {'─' * 90}")
    print(f"  {'AVERAGE':<44} {avg['der_collar0']:>7.2f}% {avg['der_collar025']:>8.2f}% "
          f"{avg['miss_rate_c0']:>6.2f}% {avg['false_alarm_c0']:>6.2f}% "
          f"{avg['confusion_c0']:>6.2f}% {avg['jer']:>6.2f}%")
    print(f"{'=' * 90}")

    # Save JSON report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_files": n,
        "global_metrics_duration_weighted": {
            "der_collar0": g_c0["der"],
            "miss_rate_c0": g_c0["miss"],
            "false_alarm_c0": g_c0["fa"],
            "confusion_c0": g_c0["conf"],
            "der_collar025": g_c025["der"],
            "miss_rate_c025": g_c025["miss"],
            "false_alarm_c025": g_c025["fa"],
            "confusion_c025": g_c025["conf"],
            "jer": g_jer,
            "total_speech_sec": g_c0["total_speech_sec"],
            "note": "Trọng số theo thời lượng + UEM — so sánh được với report.md",
        },
        "average_metrics": avg,
        "speaker_count_match": f"{speaker_match}/{n}",
        "latency_summary": {
            "avg_ms": round(np.mean(all_latencies), 2),
            "p50_ms": round(np.percentile(all_latencies, 50), 2),
            "p95_ms": round(np.percentile(all_latencies, 95), 2),
            "p99_ms": round(np.percentile(all_latencies, 99), 2),
        } if all_latencies else {},
        "per_file": summary_rows,
    }
    report_path = output_dir / "batch_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Report saved: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch evaluate diarization: stream N audio files, compute metrics vs ground truth"
    )
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="Thư mục chứa các file .wav")
    parser.add_argument("--rttm-dir", type=str, required=True,
                        help="Thư mục chứa ground truth .rttm (tên file trùng với audio)")
    parser.add_argument("--output-dir", type=str, default="eval_output",
                        help="Thư mục lưu predicted RTTM và report (default: eval_output)")
    parser.add_argument("--server", type=str, default="ws://localhost:8000/ws/diarize",
                        help="WebSocket server URL")
    parser.add_argument("--fast", action="store_true",
                        help="Gửi tốc độ tối đa, không mô phỏng real-time")
    parser.add_argument("--key", type=str, default=None,
                        help="API key nếu server yêu cầu xác thực")
    args = parser.parse_args()
    asyncio.run(run_batch(args))
