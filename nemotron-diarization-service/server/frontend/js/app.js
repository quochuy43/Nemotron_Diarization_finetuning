/**
 * Nemotron-3 Diarization Streaming Hub - Client Application
 * WebSocket Real-Time Audio Streaming (240ms chunk SLA) & NIST RTTM Evaluation
 */

let socket = null;
let audioContext = null;
let micStream = null;
let processorNode = null;
let isStreaming = false;
let inputMode = 'mic'; // 'mic' | 'file'
let selectedAudioFile = null;
let fileStreamInterval = null;

// Bộ nhớ lưu trữ tất cả các chunks đã nhận được trong phiên
let recordedChunks = [];
let currentHypothesisRttm = "";

// 1. Khởi tạo 4 Speaker slots với giao diện và màu sắc đồng bộ
function initSpeakerSlots() {
  const speakerSlotsEl = document.getElementById("speakerSlots");
  if (!speakerSlotsEl) return;
  speakerSlotsEl.innerHTML = "";
  for (let i = 0; i < 8; i++) {
    const badge = document.createElement("div");
    badge.className = `spk-badge spk-${i}`;
    badge.id = `spkBadge_${i}`;
    badge.innerHTML = `
      <div class="spk-avatar">S${i}</div>
      <div class="spk-info">
        <div class="spk-title">SPK_${String(i).padStart(2, '0')}</div>
        <div class="spk-status" id="spkStatus_${i}">Im lặng</div>
      </div>
    `;
    speakerSlotsEl.appendChild(badge);
  }
}

// 2. Chuyển đổi chế độ Mic / File
function switchInputMode(mode) {
  if (isStreaming) return;
  inputMode = mode;
  document.getElementById("tabMic").classList.toggle("active", mode === 'mic');
  document.getElementById("tabFile").classList.toggle("active", mode === 'file');
  document.getElementById("fileUploadRow").style.display = (mode === 'file') ? "block" : "none";
}

async function handleAudioFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    if (isStreaming) {
      stopStreaming();
    }
    await resetAll();
    selectedAudioFile = file;
    document.getElementById("audioFileName").innerText = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  }
}

function handleRttmFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (event) => {
      document.getElementById("gtRttmText").value = event.target.result;
      document.getElementById("rttmFileName").innerText = file.name;
      updateGtStats();
      // Nếu đã có kết quả dự đoán (currentHypothesisRttm), tự động cập nhật bảng điểm metrics luôn
      if (currentHypothesisRttm) {
        calculateMetricsNow();
      }
    };
    reader.readAsText(file);
  }
}

function updateGtStats() {
  const txt = document.getElementById("gtRttmText").value.trim();
  if (!txt) {
    document.getElementById("gtBadge").innerText = "Chưa có nhãn gốc";
    document.getElementById("gtBadge").style.color = "var(--accent-orange)";
    return;
  }
  const lines = txt.split('\n').filter(l => l.trim().startsWith("SPEAKER"));
  const spks = new Set();
  lines.forEach(l => {
    const p = l.trim().split(/\s+/);
    if (p.length >= 8) spks.add(p[7]);
  });
  document.getElementById("gtBadge").innerText = `Đã nạp: ${spks.size} người nói (${lines.length} đoạn)`;
  document.getElementById("gtBadge").style.color = "var(--accent-green)";
}

// 3. Kết nối WebSocket
function connectWebSocket(onReady) {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${proto}//${window.location.host}/ws/diarize`;
  socket = new WebSocket(wsUrl);

  socket.onopen = () => {
    document.getElementById("wsStatus").innerHTML = '<span class="status-dot"></span> ĐÃ KẾT NỐI WEBSOCKET';
    document.getElementById("wsStatus").style.color = 'var(--accent-green)';
    if (onReady) onReady();
  };

  socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'diarization_event') {
      handleDiarizationEvent(data);
    }
  };

  socket.onclose = () => {
    document.getElementById("wsStatus").innerHTML = '<span class="status-dot" style="background:#ef4444;"></span> MẤT KẾT NỐI';
    document.getElementById("wsStatus").style.color = '#ef4444';
    if (isStreaming) stopStreaming();
  };
}

// Gửi lệnh RESET lên server và đợi server xác nhận đã xóa sạch ring buffer và đưa timestamp về 0.0s
function sendResetToServer() {
  return new Promise((resolve) => {
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      resolve();
      return;
    }
    let resolved = false;
    const onResetAck = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'reset_ack') {
          resolved = true;
          socket.removeEventListener('message', onResetAck);
          resolve();
        }
      } catch (e) {}
    };
    socket.addEventListener('message', onResetAck);
    socket.send(JSON.stringify({ action: "reset" }));
    setTimeout(() => {
      if (!resolved) {
        socket.removeEventListener('message', onResetAck);
        resolve();
      }
    }, 250);
  });
}

// 4. Xử lý sự kiện nhận từ server
function handleDiarizationEvent(event) {
  recordedChunks.push(event);

  document.getElementById("valLatency").innerText = `${event.latency_ms} ms`;
  document.getElementById("streamTimeLabel").innerText = `${event.time_end_sec.toFixed(2)}s`;
  document.getElementById("chunkCountLabel").innerText = `${event.chunk_index} chunks`;
  document.getElementById("activeSpeakerCountLabel").innerText = `${event.num_active_speakers} người`;

  // Cập nhật card trạng thái loa
  const activeChans = new Set(event.active_speakers.map(s => s.channel_index));
  for (let i = 0; i < 8; i++) {
    const badge = document.getElementById(`spkBadge_${i}`);
    const statusEl = document.getElementById(`spkStatus_${i}`);
    if (activeChans.has(i)) {
      badge.classList.add("active");
      const spkInfo = event.active_speakers.find(s => s.channel_index === i);
      statusEl.innerText = `Nói (${(spkInfo.confidence * 100).toFixed(0)}%)`;
    } else {
      badge.classList.remove("active");
      statusEl.innerText = "Im lặng";
    }
  }

  // Thêm log timeline
  if (event.active_speakers.length > 0) {
    const logContainer = document.getElementById("timelineLog");
    const spkNames = event.active_speakers.map(s => s.speaker_id).join(", ");
    const firstChan = event.active_speakers[0].channel_index;

    const row = document.createElement("div");
    row.className = `log-item spk_${firstChan % 8}`;
    row.innerHTML = `
      <span>[${event.time_start_sec.toFixed(2)}s - ${event.time_end_sec.toFixed(2)}s] <strong>${spkNames}</strong></span>
      <span style="color:var(--text-muted);">${event.latency_ms}ms</span>
    `;
    logContainer.appendChild(row);
    logContainer.scrollTop = logContainer.scrollHeight;
  }
}

// 5. Bắt đầu Stream
function startStreaming() {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    connectWebSocket(() => startStreamingAction());
  } else {
    startStreamingAction();
  }
}

async function startStreamingAction() {
  // 1. Dừng interval cũ nếu còn
  if (fileStreamInterval) {
    clearInterval(fileStreamInterval);
    fileStreamInterval = null;
  }

  // 2. Gửi lệnh reset lên server để xóa sạch buffer và đưa timestamp về 0.0s
  await sendResetToServer();

  // 3. Khởi tạo lại trạng thái UI & client
  recordedChunks = [];
  currentHypothesisRttm = "";
  document.getElementById("timelineLog").innerHTML = "";
  document.getElementById("rttmOutputBox").innerText = "Chưa có kết quả RTTM.";
  document.getElementById("resultsSection").style.display = "none";
  document.getElementById("metricsScorecard").style.display = "none";
  document.getElementById("streamTimeLabel").innerText = "0.00s";
  document.getElementById("chunkCountLabel").innerText = "0 chunks";
  document.getElementById("activeSpeakerCountLabel").innerText = "0 người";
  for (let i = 0; i < 8; i++) {
    const badge = document.getElementById(`spkBadge_${i}`);
    if (badge) badge.classList.remove("active");
    const statusEl = document.getElementById(`spkStatus_${i}`);
    if (statusEl) statusEl.innerText = "Im lặng";
  }

  // 4. Khởi chạy luồng thu âm hoặc stream file
  if (inputMode === 'mic') {
    startMicCapture();
  } else {
    if (!selectedAudioFile) {
      alert("Vui lòng chọn 1 file âm thanh WAV hoặc MP3 trước khi bắt đầu!");
      return;
    }
    startFileStreaming(selectedAudioFile);
  }
}

async function startMicCapture() {
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const source = audioContext.createMediaStreamSource(micStream);

    await audioContext.audioWorklet.addModule('/static/js/audio-processor.js');
    processorNode = new AudioWorkletNode(audioContext, 'pcm16-processor');
    source.connect(processorNode);

    processorNode.port.onmessage = (e) => {
      if (!isStreaming || socket.readyState !== WebSocket.OPEN) return;
      socket.send(e.data);
    };

    isStreaming = true;
    document.getElementById("btnStart").disabled = true;
    document.getElementById("btnStop").disabled = false;
  } catch (err) {
    alert("Lỗi truy cập Microphone: " + err);
  }
}

function parseWavFile(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const tag = (o) => String.fromCharCode(view.getUint8(o), view.getUint8(o+1), view.getUint8(o+2), view.getUint8(o+3));
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null;

  let fmtOff = -1, dataOff = -1, dataSize = 0, pos = 12;
  while (pos < view.byteLength - 8) {
    const id = tag(pos);
    const sz = view.getUint32(pos + 4, true);
    if (id === 'fmt ') fmtOff = pos + 8;
    if (id === 'data') { dataOff = pos + 8; dataSize = sz; break; }
    pos += 8 + sz + (sz % 2);
  }
  if (fmtOff < 0 || dataOff < 0) return null;
  return {
    audioFormat:   view.getUint16(fmtOff, true),
    numChannels:   view.getUint16(fmtOff + 2, true),
    sampleRate:    view.getUint32(fmtOff + 4, true),
    bitsPerSample: view.getUint16(fmtOff + 14, true),
    dataOff, dataSize
  };
}

function wavPcmToFloat32Mono(arrayBuffer, info) {
  if (info.bitsPerSample !== 16 || info.audioFormat !== 1) return null;
  const samples = new Int16Array(arrayBuffer, info.dataOff, info.dataSize / 2);
  const ch = info.numChannels;
  const n = Math.floor(samples.length / ch);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = samples[i * ch] / 32768.0;
  return out;
}

async function startFileStreaming(file) {
  isStreaming = true;
  document.getElementById("btnStart").disabled = true;
  document.getElementById("btnStop").disabled = false;

  const arrayBuffer = await file.arrayBuffer();
  let floatData;
  let srcRate = 16000;

  // Parse WAV trực tiếp → tránh double-resampling qua AudioContext (48kHz → 16kHz)
  const wav = parseWavFile(arrayBuffer);
  if (wav && wav.audioFormat === 1 && wav.bitsPerSample === 16) {
    floatData = wavPcmToFloat32Mono(arrayBuffer, wav);
    srcRate = wav.sampleRate;
    console.log('[Audio] WAV direct read — sr:', srcRate, 'samples:', floatData.length);
  }

  if (!floatData) {
    // Fallback: non-WAV hoặc WAV không phải PCM16 → dùng browser decode
    const tempCtx = new (window.AudioContext || window.webkitAudioContext)();
    const decoded = await tempCtx.decodeAudioData(arrayBuffer);
    srcRate = decoded.sampleRate;
    floatData = decoded.numberOfChannels > 1
      ? decoded.getChannelData(0)
      : decoded.getChannelData(0);
    console.log('[Audio] Browser decode fallback — sr:', srcRate, 'samples:', floatData.length);
    tempCtx.close();
  }

  // Resample sang 16kHz chỉ khi cần (1 lần duy nhất, từ đúng sample rate gốc)
  if (srcRate !== 16000) {
    const srcBuf = new AudioBuffer({ length: floatData.length, sampleRate: srcRate, numberOfChannels: 1 });
    srcBuf.copyToChannel(floatData, 0);
    const offCtx = new OfflineAudioContext(1, Math.ceil(floatData.length * 16000 / srcRate), 16000);
    const src = offCtx.createBufferSource();
    src.buffer = srcBuf;
    src.connect(offCtx.destination);
    src.start();
    const resampled = await offCtx.startRendering();
    floatData = resampled.getChannelData(0);
    console.log('[Audio] Resampled', srcRate, '→ 16000, samples:', floatData.length);
  }

  const chunkSamples = 3840; // 240ms @ 16kHz
  let offset = 0;
  fileStreamInterval = setInterval(() => {
    if (!isStreaming || socket.readyState !== WebSocket.OPEN) {
      clearInterval(fileStreamInterval);
      fileStreamInterval = null;
      return;
    }

    if (offset >= floatData.length) {
      clearInterval(fileStreamInterval);
      fileStreamInterval = null;
      setTimeout(() => {
        if (isStreaming) stopStreaming();
      }, 350);
      return;
    }

    const slice = floatData.subarray(offset, offset + chunkSamples);
    const pcm16 = new Int16Array(slice.length);
    for (let i = 0; i < slice.length; i++) {
      const s = Math.max(-1, Math.min(1, slice[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    socket.send(pcm16.buffer);
    offset += chunkSamples;
  }, 240);
}

// 6. Dừng Thu Âm & Tạo File RTTM Tức Thì
function stopStreaming() {
  isStreaming = false;
  if (fileStreamInterval) {
    clearInterval(fileStreamInterval);
    fileStreamInterval = null;
  }
  if (processorNode) {
    try { processorNode.disconnect(); } catch (e) {}
    processorNode = null;
  }
  if (micStream) {
    try { micStream.getTracks().forEach(t => t.stop()); } catch (e) {}
    micStream = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
  }

  document.getElementById("btnStart").disabled = false;
  document.getElementById("btnStop").disabled = true;

  // Trả lại trạng thái các loa về Im lặng
  for (let i = 0; i < 8; i++) {
    const badge = document.getElementById(`spkBadge_${i}`);
    if (badge) badge.classList.remove("active");
    const statusEl = document.getElementById(`spkStatus_${i}`);
    if (statusEl) statusEl.innerText = "Im lặng";
  }
  document.getElementById("activeSpeakerCountLabel").innerText = "0 người";

  // Sinh file RTTM từ các chunks đã lưu
  generateAndDisplayRttm();
}

function generateAndDisplayRttm() {
  if (recordedChunks.length === 0) {
    alert("Chưa ghi nhận được âm thanh nào để xuất RTTM!");
    return;
  }

  // Thuật toán gộp các chunk liên tục của cùng speaker
  const speakerIntervals = {};
  recordedChunks.forEach(chunk => {
    chunk.active_speakers.forEach(spk => {
      const sid = spk.speaker_id;
      if (!speakerIntervals[sid]) speakerIntervals[sid] = [];
      speakerIntervals[sid].push({ start: chunk.time_start_sec, end: chunk.time_end_sec });
    });
  });

  const rttmLines = [];
  const uri = selectedAudioFile ? selectedAudioFile.name.replace(/\.[^/.]+$/, "") : "live_stream_session";

  Object.keys(speakerIntervals).sort().forEach(spkId => {
    const intervals = speakerIntervals[spkId].sort((a, b) => a.start - b.start);
    let curStart = null;
    let curEnd = null;

    intervals.forEach(it => {
      if (curStart === null) {
        curStart = it.start;
        curEnd = it.end;
      } else {
        // Nối khoảng ngắt nhỏ hơn 0.15s (min_gap_off)
        if (it.start - curEnd <= 0.16) {
          curEnd = it.end;
        } else {
          const dur = curEnd - curStart;
          if (dur >= 0.05) {
            rttmLines.push(`SPEAKER ${uri} 1 ${curStart.toFixed(2)} ${dur.toFixed(2)} <NA> <NA> ${spkId} <NA> <NA>`);
          }
          curStart = it.start;
          curEnd = it.end;
        }
      }
    });

    if (curStart !== null) {
      const dur = curEnd - curStart;
      if (dur >= 0.05) {
        rttmLines.push(`SPEAKER ${uri} 1 ${curStart.toFixed(2)} ${dur.toFixed(2)} <NA> <NA> ${spkId} <NA> <NA>`);
      }
    }
  });

  // Sắp xếp theo mốc thời gian bắt đầu
  rttmLines.sort((a, b) => parseFloat(a.split(/\s+/)[3]) - parseFloat(b.split(/\s+/)[3]));

  currentHypothesisRttm = rttmLines.join("\n");
  document.getElementById("rttmOutputBox").innerText = currentHypothesisRttm || "Không có đoạn phát biểu nào được ghi nhận.";
  document.getElementById("resultsSection").style.display = "flex";

  // Nếu có Ground Truth RTTM, tự động tính toán metrics luôn
  const gtText = document.getElementById("gtRttmText").value.trim();
  if (gtText && currentHypothesisRttm) {
    calculateMetricsNow();
  } else {
    document.getElementById("noGtNotice").style.display = "flex";
    document.getElementById("metricsScorecard").style.display = "none";
  }
}

// 7. Gọi API Chấm Điểm Metrics NIST với Ground Truth
async function calculateMetricsNow() {
  const gtText = document.getElementById("gtRttmText").value.trim();
  if (!gtText) {
    alert("Vui lòng tải lên hoặc dán nội dung Ground Truth RTTM trước!");
    return;
  }
  if (!currentHypothesisRttm) {
    alert("Chưa có kết quả RTTM dự đoán!");
    return;
  }

  try {
    const resp = await fetch("/api/evaluate_rttm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reference_rttm: gtText,
        hypothesis_rttm: currentHypothesisRttm
      })
    });
    const res = await resp.json();

    if (res.status === "success") {
      const m = res.metrics;
      document.getElementById("valDerC0").innerText = `${m.der_collar0.toFixed(2)}%`;
      document.getElementById("valDerC0Sub").innerText = `Miss: ${m.miss_rate_c0}% | FA: ${m.false_alarm_c0}% | Conf: ${m.confusion_c0}%`;
      document.getElementById("valDerC025").innerText = `${m.der_collar025.toFixed(2)}%`;
      document.getElementById("valJer").innerText = `${m.jer.toFixed(2)}%`;
      document.getElementById("valSca").innerText = m.speaker_count_match ? "✅ CHÍNH XÁC" : "⚠️ SAI LỆCH";
      document.getElementById("valScaSub").innerText = `Ref: ${m.ref_speakers} spks | Hyp: ${m.hyp_speakers} spks`;

      document.getElementById("metricsScorecard").style.display = "flex";
      document.getElementById("noGtNotice").style.display = "none";
    } else {
      alert("Lỗi tính toán metrics: " + res.message);
    }
  } catch (err) {
    alert("Không thể kết nối đến API tính toán metrics: " + err);
  }
}

function copyRttmToClipboard() {
  if (!currentHypothesisRttm) return;
  navigator.clipboard.writeText(currentHypothesisRttm).then(() => {
    alert("Đã sao chép nội dung RTTM vào bộ nhớ đệm!");
  });
}

function downloadRttmFile() {
  if (!currentHypothesisRttm) return;
  const blob = new Blob([currentHypothesisRttm], { type: "text/plain;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "hypothesis_output.rttm";
  a.click();
}

async function resetAll() {
  if (isStreaming) {
    isStreaming = false;
    document.getElementById("btnStart").disabled = false;
    document.getElementById("btnStop").disabled = true;
  }
  if (fileStreamInterval) {
    clearInterval(fileStreamInterval);
    fileStreamInterval = null;
  }
  if (processorNode) {
    try { processorNode.disconnect(); } catch (e) {}
    processorNode = null;
  }
  if (micStream) {
    try { micStream.getTracks().forEach(t => t.stop()); } catch (e) {}
    micStream = null;
  }
  if (audioContext && audioContext.state !== 'closed') {
    try { audioContext.close(); } catch (e) {}
    audioContext = null;
  }

  // Xóa sạch dữ liệu client
  recordedChunks = [];
  currentHypothesisRttm = "";
  document.getElementById("timelineLog").innerHTML = `
    <div style="color: var(--text-muted); text-align: center; margin-top: 40px;">
      Bấm 'Bắt Đầu Thu' để xem sự kiện phân đoạn...
    </div>
  `;
  document.getElementById("rttmOutputBox").innerText = "Chưa có kết quả RTTM.";
  document.getElementById("resultsSection").style.display = "none";
  document.getElementById("metricsScorecard").style.display = "none";
  document.getElementById("streamTimeLabel").innerText = "0.00s";
  document.getElementById("chunkCountLabel").innerText = "0 chunks";
  document.getElementById("valLatency").innerText = "-- ms";
  document.getElementById("activeSpeakerCountLabel").innerText = "0 người";
  for (let i = 0; i < 8; i++) {
    const badge = document.getElementById(`spkBadge_${i}`);
    if (badge) badge.classList.remove("active");
    const statusEl = document.getElementById(`spkStatus_${i}`);
    if (statusEl) statusEl.innerText = "Im lặng";
  }

  // Gửi reset lên server
  await sendResetToServer();
}

window.onload = () => {
  initSpeakerSlots();
  connectWebSocket();
};
