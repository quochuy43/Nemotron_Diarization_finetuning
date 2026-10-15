#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
Script: run_train_grid.py (Giai đoạn 5)
Mục đích: Huấn luyện tinh chỉnh (Fine-tuning) mô hình Nemotron-3 Diarization 0.32s
          theo 4 phiên bản của ma trận thí nghiệm (v1.0, v1.1, v1.2, v1.3).
Đặc tính kỹ thuật (Chuẩn theo plan.md & README.md):
  - Cấu hình Streaming Ultra-Low Latency 0.32s (chunk 3, right context 1, spkcache 264).
  - Đóng băng nửa dưới 16/31 lớp Transformer Encoder để bảo toàn acoustic features.
  - Tổ chức lấy mẫu phân tầng & Loss Permutation-Invariant (Hungarian PIT-BCE).
  - Chiến lược lưu ~50 Checkpoints đều đặn xuyên suốt quá trình huấn luyện.
  - Đánh giá tự động định kỳ trên tập Dev Set (43 files) theo đa lát cắt (Tổng, 2 người, 3-7 người).
  - Theo dõi đường cong hội tụ (Convergence Log) và tự động xếp hạng Top-5 Checkpoints.
===============================================================================
"""

import os
import sys
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Đưa thư mục cha và thư mục 02_baseline_eval vào sys.path để tái sử dụng module chấm điểm
current_dir = Path(__file__).resolve().parent
scripts_dir = current_dir.parent
baseline_eval_dir = scripts_dir / "02_baseline_eval"
if str(baseline_eval_dir) not in sys.path:
    sys.path.insert(0, str(baseline_eval_dir))

from evaluate_metrics import parse_rttm_file, compute_metrics_for_dataset, resolve_file_path
from tune_postprocessing import binarize_and_filter_probs, write_segments_to_rttm


# ===============================================================================
# 1. DATASET & DATALOADER CHO HUẤN LUYỆN DIARIZATION
# ===============================================================================

class NemotronDiarizationDataset(Dataset):
    """
    Dataset nạp audio WAV 16kHz và chuyển đổi nhãn RTTM thành frame targets đa nhãn (T, 8).
    """
    def __init__(
        self,
        manifest_path: str,
        max_duration_seconds: float = 30.0,
        frame_step: float = 0.01,
        max_speakers: int = 8,
        sample_rate: int = 16000,
        limit_samples: Optional[int] = None,
        is_training: bool = True
    ):
        self.max_duration = max_duration_seconds
        self.frame_step = frame_step
        self.max_speakers = max_speakers
        self.sample_rate = sample_rate
        self.is_training = is_training
        self.items = []

        manifest_file = Path(resolve_file_path(manifest_path))
        if not manifest_file.exists():
            raise FileNotFoundError(f"Không tìm thấy manifest tại: {manifest_path}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.items.append(json.loads(line))

        if limit_samples is not None and limit_samples > 0:
            self.items = self.items[:limit_samples]

    def __len__(self) -> int:
        return len(self.items)

    def _parse_rttm_to_frames(
        self, rttm_path: str, start_time: float, duration: float, num_frames: int
    ) -> np.ndarray:
        """Chuyển đổi nhãn RTTM thành ma trận nhị phân (num_frames, max_speakers)."""
        targets = np.zeros((num_frames, self.max_speakers), dtype=np.float32)
        if not os.path.exists(rttm_path):
            return targets

        spk_to_idx = {}
        with open(rttm_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    onset = float(parts[3])
                    dur = float(parts[4])
                    offset = onset + dur
                    spk_name = parts[7]

                    # Gán chỉ số speaker
                    if spk_name not in spk_to_idx:
                        if len(spk_to_idx) < self.max_speakers:
                            spk_to_idx[spk_name] = len(spk_to_idx)
                        else:
                            continue

                    spk_id = spk_to_idx[spk_name]

                    # Giao thoa với khoảng cắt âm thanh [start_time, start_time + duration]
                    seg_start = max(0.0, onset - start_time)
                    seg_end = min(duration, offset - start_time)

                    if seg_end > seg_start:
                        f_start = int(round(seg_start / self.frame_step))
                        f_end = int(round(seg_end / self.frame_step))
                        f_start = max(0, min(num_frames - 1, f_start))
                        f_end = max(0, min(num_frames, f_end))
                        targets[f_start:f_end, spk_id] = 1.0

        return targets

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        wav_path = resolve_file_path(item["audio_filepath"])
        rttm_path = resolve_file_path(item["rttm_filepath"])
        total_duration = float(item.get("duration", 0.0))

        # Đọc thông tin file âm thanh
        info = sf.info(wav_path)
        sr = info.samplerate
        total_frames = info.frames

        # Cắt ngẫu nhiên 1 đoạn nếu audio dài hơn max_duration
        target_samples = int(self.max_duration * self.sample_rate)
        if total_frames > target_samples and self.is_training:
            max_start_frame = total_frames - target_samples
            start_frame = random.randint(0, max_start_frame)
            start_sec = start_frame / sr
            actual_dur = self.max_duration
            audio_data, _ = sf.read(wav_path, start=start_frame, stop=start_frame + target_samples)
        else:
            start_sec = 0.0
            actual_dur = min(total_duration, self.max_duration)
            audio_data, _ = sf.read(wav_path, stop=int(actual_dur * sr))

        # Đảm bảo mono và float32
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
        audio_data = audio_data.astype(np.float32)

        # Tính số lượng frame mục tiêu tương ứng
        num_frames = int(round(len(audio_data) / sr / self.frame_step)) + 1
        targets = self._parse_rttm_to_frames(rttm_path, start_sec, actual_dur, num_frames)

        return {
            "audio": audio_data,
            "targets": targets,
            "audio_path": wav_path,
            "duration": actual_dur
        }


def collate_diarization_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gộp các mẫu trong batch và đệm độ dài (Padding)."""
    max_audio_len = max(len(b["audio"]) for b in batch)
    max_target_len = max(len(b["targets"]) for b in batch)
    max_speakers = batch[0]["targets"].shape[1]

    padded_audio = []
    padded_targets = []
    audio_lengths = []
    target_lengths = []

    for b in batch:
        a = b["audio"]
        t = b["targets"]
        audio_lengths.append(len(a))
        target_lengths.append(len(t))

        # Pad audio với 0
        if len(a) < max_audio_len:
            a_pad = np.pad(a, (0, max_audio_len - len(a)), mode="constant")
        else:
            a_pad = a
        padded_audio.append(a_pad)

        # Pad targets với 0
        if len(t) < max_target_len:
            t_pad = np.pad(t, ((0, max_target_len - len(t)), (0, 0)), mode="constant")
        else:
            t_pad = t
        padded_targets.append(t_pad)

    return {
        "audio": torch.from_numpy(np.stack(padded_audio)).float(),
        "targets": torch.from_numpy(np.stack(padded_targets)).float(),
        "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long)
    }


# ===============================================================================
# 2. HÀM TỔN THẤT PERMUTATION-INVARIANT TRAINING (PIT-BCE)
# ===============================================================================

class PermutationInvariantBCELoss(nn.Module):
    """
    Hàm mất mát hoán vị bất biến (PIT):
    Sử dụng thuật toán Hungarian (Linear Sum Assignment) để ghép cặp tối ưu
    giữa 8 kênh dự đoán và 8 kênh nhãn thực tế trước khi tính Binary Cross Entropy.
    """
    def __init__(self, pos_weight: float = 1.5):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, T_logits, S)
        targets: (B, T_targets, S)
        """
        B, T_l, S = logits.shape
        T_t = targets.shape[1]
        device = logits.device

        # Cân bằng độ dài frame giữa logits và targets
        if T_l != T_t:
            if T_l > T_t:
                logits = logits[:, :T_t, :]
            else:
                targets = targets[:, :T_l, :]

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        batch_losses = []

        for b in range(B):
            sample_logits = logits[b]    # (T, S)
            sample_targets = targets[b]  # (T, S)

            # Tính ma trận chi phí (Cost Matrix S x S)
            cost_matrix = np.zeros((S, S), dtype=np.float32)
            with torch.no_grad():
                probs = torch.sigmoid(sample_logits)
                for i in range(S):
                    for j in range(S):
                        # Chi phí = BCE không gradient
                        cost = -(sample_targets[:, j] * torch.log(probs[:, i] + 1e-7) +
                                 (1.0 - sample_targets[:, j]) * torch.log(1.0 - probs[:, i] + 1e-7)).mean().item()
                        cost_matrix[i, j] = cost

            # Giải Hungarian matching để tìm phép hoán vị tối ưu
            pred_idx, target_idx = linear_sum_assignment(cost_matrix)

            # Sắp xếp lại logits theo phép hoán vị tối ưu
            permuted_logits = sample_logits[:, pred_idx]
            matched_targets = sample_targets[:, target_idx]

            # Tính BCE có trọng số cho mẫu này
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                permuted_logits,
                matched_targets,
                pos_weight=self.pos_weight.to(device)
            )
            batch_losses.append(bce_loss)

        return torch.stack(batch_losses).mean()


# ===============================================================================
# 3. QUẢN LÝ MÔ HÌNH & ĐÓNG BĂNG TẦNG (LAYER FREEZING)
# ===============================================================================

def build_model_and_freeze_encoder(
    pretrained_model_path: str = "nvidia/Nemotron-3-Diarization",
    freeze_layers: int = 16,
    device: str = "cuda"
) -> Tuple[Any, Any]:
    """
    Nạp mô hình Nemotron-3 Diarization và đóng băng nửa dưới (16 lớp đầu) của Encoder.
    """
    from transformers import AutoProcessor, AutoModelForAudioFrameClassification

    print(f"\n[*] Đang nạp mô hình gốc: {pretrained_model_path}...")
    processor = AutoProcessor.from_pretrained(pretrained_model_path)
    model = AutoModelForAudioFrameClassification.from_pretrained(pretrained_model_path)

    # Đóng băng 16 lớp encoder đầu tiên theo đúng thiết kế plan.md
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        should_freeze = False
        # Kiểm tra lớp thuộc 16 lớp đầu (0 đến freeze_layers - 1)
        for layer_idx in range(freeze_layers):
            if f"layers.{layer_idx}." in name:
                should_freeze = True
                break

        if should_freeze:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trainable_params += param.numel()

    model = model.to(device)
    total_params = frozen_params + trainable_params
    freeze_pct = (frozen_params / total_params) * 100

    print("=" * 80)
    print("🔒 CHIẾN LƯỢC ĐÓNG BĂNG TẦNG (LAYER FREEZING) HOÀN TẤT:")
    print(f"  [+] Tổng số tham số (Total):     {total_params:,}")
    print(f"  [+] Đã đóng băng (Frozen):       {frozen_params:,} ({freeze_pct:.1f}%) [Layers 0 -> {freeze_layers-1}]")
    print(f"  [+] Cho phép huấn luyện (Train): {trainable_params:,} ({100-freeze_pct:.1f}%) [Upper Layers & Head]")
    print("=" * 80)

    return model, processor


# ===============================================================================
# 4. HÀM ĐÁNH GIÁ NHANH TRÊN DEV SET (DEV EVALUATOR)
# ===============================================================================

def evaluate_checkpoint_on_dev(
    model: Any,
    processor: Any,
    dev_manifest: str,
    device: str = "cuda",
    limit_files: Optional[int] = None,
    threshold: float = 0.45,
    min_duration_on: float = 0.05,
    min_duration_off: float = 0.15
) -> Dict[str, float]:
    """
    Đánh giá mô hình tại checkpoint hiện tại trên tập Dev:
    Đo DER Collar 0s/0.25s, JER, SCA cho toàn bộ tập và các lát cắt.
    """
    model.eval()
    temp_dir = current_dir / "temp_eval"
    temp_rttm_dir = temp_dir / "rttm"
    temp_rttm_dir.mkdir(parents=True, exist_ok=True)

    dev_manifest_p = resolve_file_path(dev_manifest)
    manifest_items = []
    with open(dev_manifest_p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                manifest_items.append(json.loads(line))

    if limit_files is not None and limit_files > 0:
        manifest_items = manifest_items[:limit_files]

    with torch.inference_mode():
        for item in manifest_items:
            wav_path = resolve_file_path(item["audio_filepath"])
            stem = Path(wav_path).stem
            wav_data, sr = sf.read(wav_path)
            if wav_data.ndim > 1:
                wav_data = wav_data[:, 0]

            inputs = processor(wav_data, sampling_rate=16000, return_tensors="pt").to(device)
            logits = model(**inputs).logits # (1, num_frames, 8)
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

            # Hậu xử lý ra segments RTTM
            segs = binarize_and_filter_probs(
                probs,
                frame_step=0.01,
                threshold=threshold,
                min_duration_on=min_duration_on,
                min_duration_off=min_duration_off
            )
            out_rttm = temp_rttm_dir / f"{stem}.rttm"
            write_segments_to_rttm(segs, uri=stem, rttm_path=str(out_rttm))

    # Tính toán chỉ số NIST
    metrics = compute_metrics_for_dataset(
        manifest_path=str(dev_manifest_p),
        hyp_rttm_dir=str(temp_rttm_dir),
        collar_main=0.0,
        collar_secondary=0.25,
        verbose=False
    )

    # Dọn dẹp tạm thời
    for f in temp_rttm_dir.glob("*.rttm"):
        try:
            f.unlink()
        except Exception:
            pass

    return {
        "der_collar0": metrics["overall"]["der_collar0"],
        "miss_rate_c0": metrics["overall"].get("miss_rate_c0", 0.0),
        "false_alarm_c0": metrics["overall"].get("false_alarm_c0", 0.0),
        "confusion_c0": metrics["overall"].get("confusion_c0", 0.0),

        "der_collar025": metrics["overall"]["der_collar025"],
        "miss_rate_c025": metrics["overall"].get("miss_rate_c025", 0.0),
        "false_alarm_c025": metrics["overall"].get("false_alarm_c025", 0.0),
        "confusion_c025": metrics["overall"].get("confusion_c025", 0.0),

        "der_2spk_c0": metrics["slice_2spk_real"]["der_collar0"],
        "miss_2spk_c0": metrics["slice_2spk_real"].get("miss_rate_c0", 0.0),
        "fa_2spk_c0": metrics["slice_2spk_real"].get("false_alarm_c0", 0.0),
        "conf_2spk_c0": metrics["slice_2spk_real"].get("confusion_c0", 0.0),

        "der_3to7spk_c0": metrics["slice_3to7spk_sim"]["der_collar0"],
        "miss_3to7spk_c0": metrics["slice_3to7spk_sim"].get("miss_rate_c0", 0.0),
        "fa_3to7spk_c0": metrics["slice_3to7spk_sim"].get("false_alarm_c0", 0.0),
        "conf_3to7spk_c0": metrics["slice_3to7spk_sim"].get("confusion_c0", 0.0),

        "jer": metrics["overall"]["jer"],
        "sca_pct": metrics["overall"]["sca_pct"],
        "mae": metrics["overall"]["mae"]
    }


# ===============================================================================
# 5. TIẾN TRÌNH HUẤN LUYỆN CHÍNH (TRAINING & 50 CHECKPOINTS GRID SEARCH)
# ===============================================================================

def run_training_grid(
    config_path: str = "scripts/03_training/configs/sortformer_streaming_032s.yaml",
    version: str = "v1_1",
    override_manifest: Optional[str] = None,
    output_dir: Optional[str] = None,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    batch_size: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dry_run: bool = False,
    limit_train_samples: Optional[int] = None,
    limit_dev_files: Optional[int] = None
):
    """
    Quy trình huấn luyện và theo dõi 50 checkpoints theo đúng chuẩn Roadmap.
    """
    # Nạp file cấu hình YAML
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Xác định phiên bản thử nghiệm
    ver_key = version.replace(".", "_")
    if ver_key not in cfg["experiment_versions"]:
        raise ValueError(f"Phiên bản {version} không hợp lệ! Chọn trong: {list(cfg['experiment_versions'].keys())}")

    ver_cfg = cfg["experiment_versions"][ver_key]
    train_manifest = resolve_file_path(override_manifest or ver_cfg["manifest"])
    dev_manifest = resolve_file_path(cfg["validation"]["manifest"])

    total_epochs = epochs or ver_cfg["epochs"]
    lr = learning_rate or float(cfg["training"]["optimizer"]["learning_rate"])
    b_size = batch_size or int(cfg["training"]["batch_size"])
    grad_accum = int(cfg["training"]["gradient_accumulation_steps"])
    freeze_layers = int(cfg["model"]["freeze_encoder_layers"])
    target_cps = int(cfg["training"]["target_checkpoints_per_run"])

    # Thư mục lưu kết quả & checkpoints
    out_dir = Path(output_dir or f"experiments/checkpoints/{ver_cfg['name']}")
    cp_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    cp_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN GRID SEARCH: PHIÊN BẢN {version.upper()} ({ver_cfg['name']})")
    print("=" * 80)
    print(f"[*] Mô tả:             {ver_cfg['description']}")
    print(f"[*] Tỷ lệ Thật:Phỏng:  {ver_cfg['ratio_real_to_sim']}")
    print(f"[*] Train Manifest:    {train_manifest}")
    print(f"[*] Dev Manifest:      {dev_manifest}")
    print(f"[*] Tổng số Epochs:    {total_epochs}")
    print(f"[*] Learning Rate:     {lr}")
    print(f"[*] Batch Size:        {b_size} (Grad Accum: {grad_accum})")
    print(f"[*] Thiết bị (Device): {device}")
    print(f"[*] Thư mục đầu ra:    {out_dir}")
    if dry_run:
        print("[!] CHẾ ĐỘ TEST NHANH (DRY RUN): 3 bước train, 2 files dev.")

    # 1. Khởi tạo Dataset & DataLoader
    train_dataset = NemotronDiarizationDataset(
        manifest_path=train_manifest,
        max_duration_seconds=float(cfg["training"]["max_audio_length_seconds"]),
        frame_step=float(cfg["model"]["frame_step"]),
        max_speakers=int(cfg["model"]["max_speakers"]),
        sample_rate=int(cfg["model"]["sample_rate"]),
        limit_samples=limit_train_samples if not dry_run else 8,
        is_training=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=b_size if not dry_run else 2,
        shuffle=True,
        collate_fn=collate_diarization_batch,
        num_workers=0 if sys.platform == "win32" else int(cfg["training"]["num_workers"]),
        drop_last=False
    )

    steps_per_epoch = len(train_loader)
    total_training_steps = steps_per_epoch * total_epochs
    eval_interval = max(1, total_training_steps // target_cps) if not dry_run else 2

    print(f"[+] Tổng số mẫu huấn luyện: {len(train_dataset):,} mẫu.")
    print(f"[+] Số bước/epoch:          {steps_per_epoch} bước.")
    print(f"[+] Tổng số bước toàn run:  {total_training_steps} bước.")
    print(f"[+] Chu kỳ lưu checkpoint:  Mỗi {eval_interval} bước (Mục tiêu ~{target_cps} CPs).")

    # 2. Khởi tạo Mô hình & Optimizer
    model, processor = build_model_and_freeze_encoder(
        pretrained_model_path=cfg["model"]["pretrained_model_name_or_path"],
        freeze_layers=freeze_layers,
        device=device
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=float(cfg["training"]["optimizer"]["weight_decay"])
    )

    loss_fn = PermutationInvariantBCELoss(pos_weight=float(cfg["training"]["loss"]["pos_weight"]))
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # 3. Vòng lặp huấn luyện & Quét Checkpoint
    global_step = 0
    best_dev_der = float("inf")
    convergence_log = []
    top_checkpoints = []

    print("\n[*] Bắt đầu tiến trình Training Loop...")
    start_time = time.time()

    for epoch in range(1, total_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}", unit="batch")

        for step_idx, batch in enumerate(pbar):
            global_step += 1
            audio_batch = batch["audio"].to(device)
            targets_batch = batch["targets"].to(device)

            # Chuẩn bị inputs cho model (dạng list các mảng 1D cho processor)
            raw_audio_list = [a.numpy() for a in audio_batch.cpu()]
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                inputs = processor(raw_audio_list, sampling_rate=16000, return_tensors="pt").to(device)
                logits = model(**inputs).logits
                loss = loss_fn(logits, targets_batch)
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if global_step % grad_accum == 0 or global_step == total_training_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            pbar.set_postfix({"Loss": f"{loss.item() * grad_accum:.4f}", "Step": global_step})

            # Đánh giá & Lưu Checkpoint định kỳ (~50 checkpoints)
            if global_step % eval_interval == 0 or global_step == total_training_steps or dry_run:
                cp_id = len(convergence_log) + 1
                print(f"\n\n[*] Đang đánh giá Checkpoint #{cp_id} (Step {global_step}/{total_training_steps}) trên tập Dev...")

                dev_metrics = evaluate_checkpoint_on_dev(
                    model=model,
                    processor=processor,
                    dev_manifest=dev_manifest,
                    device=device,
                    limit_files=limit_dev_files if not dry_run else 2
                )

                cur_der = dev_metrics["der_collar0"]
                record = {
                    "checkpoint_id": cp_id,
                    "step": global_step,
                    "epoch": epoch,
                    "loss": round(loss.item() * grad_accum, 4),
                    "der_collar0": cur_der,
                    "miss_rate_c0": dev_metrics["miss_rate_c0"],
                    "false_alarm_c0": dev_metrics["false_alarm_c0"],
                    "confusion_c0": dev_metrics["confusion_c0"],
                    "der_collar025": dev_metrics["der_collar025"],
                    "miss_rate_c025": dev_metrics["miss_rate_c025"],
                    "false_alarm_c025": dev_metrics["false_alarm_c025"],
                    "confusion_c025": dev_metrics["confusion_c025"],
                    "der_2spk_c0": dev_metrics["der_2spk_c0"],
                    "der_3to7spk_c0": dev_metrics["der_3to7spk_c0"],
                    "jer": dev_metrics["jer"],
                    "sca_pct": dev_metrics["sca_pct"],
                    "mae": dev_metrics["mae"],
                    "checkpoint_file": f"checkpoint_{cp_id:03d}_step_{global_step}.pt"
                }
                convergence_log.append(record)

                # Lưu checkpoint file
                cp_file_path = cp_dir / record["checkpoint_file"]
                torch.save({
                    "step": global_step,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": dev_metrics,
                    "version": version
                }, cp_file_path)

                print(f"  [+] Đạt Dev DER (Collar 0s): {cur_der:.2f}% (Miss: {dev_metrics['miss_rate_c0']:.2f}% | FA: {dev_metrics['false_alarm_c0']:.2f}% | Conf: {dev_metrics['confusion_c0']:.2f}%) | 2spk: {dev_metrics['der_2spk_c0']:.2f}% | 3-7spk: {dev_metrics['der_3to7spk_c0']:.2f}%")
                print(f"  [+] Đã lưu checkpoint tại: {cp_file_path}")

                # Cập nhật danh sách Top-5 Checkpoints
                top_checkpoints = sorted(convergence_log, key=lambda x: x["der_collar0"])[:5]

                # Lưu log hội tụ ra file
                with open(out_dir / "convergence_log.json", "w", encoding="utf-8") as f:
                    json.dump(convergence_log, f, ensure_ascii=False, indent=2)

                with open(out_dir / "top_checkpoints.json", "w", encoding="utf-8") as f:
                    json.dump(top_checkpoints, f, ensure_ascii=False, indent=2)

                model.train()

            if dry_run and global_step >= 3:
                print("\n[!] Hoàn tất Dry-run thành công!")
                break

        if dry_run and global_step >= 3:
            break

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 HOÀN TẤT HUẤN LUYỆN PHIÊN BẢN {version.upper()} TRONG {elapsed/60:.2f} PHÚT!")
    print("=" * 80)
    print("🏆 TOP 5 CHECKPOINTS TỐT NHẤT TRÊN TẬP DEV:")
    for rank, cp in enumerate(top_checkpoints[:5], 1):
        miss_s = cp.get('miss_rate_c0', 0.0)
        fa_s = cp.get('false_alarm_c0', 0.0)
        conf_s = cp.get('confusion_c0', 0.0)
        print(f"  #{rank}: {cp['checkpoint_file']} (Step {cp['step']}) -> DER Tổng: {cp['der_collar0']}% [Miss: {miss_s:.2f}% | FA: {fa_s:.2f}% | Conf: {conf_s:.2f}%] | 2spk: {cp['der_2spk_c0']}% | 3-7spk: {cp['der_3to7spk_c0']}%")
    print("=" * 80)

    return top_checkpoints, convergence_log


# ===============================================================================
# 6. ĐIỂM VÀO DÒNG LỆNH CLI
# ===============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Grid Search Nemotron-3 Diarization 0.32s")
    parser.add_argument("--config", type=str, default="scripts/03_training/configs/sortformer_streaming_032s.yaml", help="Đường dẫn file YAML")
    parser.add_argument("--version", type=str, default="v1_0", choices=["v1_0", "v1.0", "v1_1", "v1.1", "v1_2", "v1.2", "v1_3", "v1.3"], help="Phiên bản huấn luyện")
    parser.add_argument("--manifest", type=str, default=None, help="Ghi đè train manifest")
    parser.add_argument("--output_dir", type=str, default=None, help="Thư mục xuất checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Số epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Thiết bị (cuda/cpu)")
    parser.add_argument("--dry_run", action="store_true", help="Chạy kiểm thử nhanh không tốn tài nguyên")
    parser.add_argument("--limit_train_samples", type=int, default=None, help="Giới hạn số mẫu train")
    parser.add_argument("--limit_dev_files", type=int, default=None, help="Giới hạn số file đánh giá dev")

    args = parser.parse_args()

    run_training_grid(
        config_path=args.config,
        version=args.version,
        override_manifest=args.manifest,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        dry_run=args.dry_run,
        limit_train_samples=args.limit_train_samples,
        limit_dev_files=args.limit_dev_files
    )
