#!/usr/bin/env python3
"""Distill a 600M teacher (XTTS-v2) into a 180M student with LoRA."""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
import lightning as pl
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DDPStrategy

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural.student import StudentTTS
from neural.speaker_encoder import SpeakerEncoder
from neural.distillation_wrapper import DistillationTrainer
from training.common import count_parameters, setup_logging, get_dataloader, get_checkpoint_callbacks


class TTSDataset(Dataset):
    """Minimal streaming-compatible dataset stub. Replace with real LibriTTS/VoxCeleb loader."""

    def __init__(self, data_dir: str, max_samples: int = 100_000):
        self.data_dir = Path(data_dir)
        self.samples = list(self.data_dir.glob("*.pt"))[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Expected .pt dict: {text_tokens, mel, speaker_waveform, teacher_mel}
        data = torch.load(self.samples[idx], weights_only=True)
        return {
            "text_tokens": data["text_tokens"],
            "mel": data["mel"],
            "speaker_waveform": data.get("speaker_waveform", torch.zeros(1, 48_000)),
            "teacher_mel": data.get("teacher_mel", data["mel"]),
        }


def collate_fn(batch):
    # Pad text tokens to max sequence length
    max_text_len = max(b["text_tokens"].size(0) for b in batch)
    text_tokens = torch.stack([
        torch.cat([b["text_tokens"], torch.zeros(max_text_len - b["text_tokens"].size(0), dtype=torch.long)])
        for b in batch
    ])

    # Pad mel spectrograms to max time dimension
    max_mel_len = max(b["mel"].size(-1) for b in batch)
    mel = torch.stack([
        torch.nn.functional.pad(b["mel"], (0, max_mel_len - b["mel"].size(-1)))
        for b in batch
    ])

    # Pad teacher mel spectrograms to max time dimension
    max_teacher_len = max(b["teacher_mel"].size(-1) for b in batch)
    teacher_mel = torch.stack([
        torch.nn.functional.pad(b["teacher_mel"], (0, max_teacher_len - b["teacher_mel"].size(-1)))
        for b in batch
    ])

    speaker_waveforms = [b["speaker_waveform"] for b in batch]
    return {
        "text_tokens": text_tokens,
        "mel": mel,
        "teacher_mel": teacher_mel,
        "speaker_waveforms": speaker_waveforms,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Student TTS via distillation")
    parser.add_argument("--data_dir", required=True, help="Directory containing preprocessed .pt files")
    parser.add_argument("--teacher_model", default="tts_models/multilingual/multi-dataset/xtts_v2", help="TTS model name for teacher")
    parser.add_argument("--output_dir", default="checkpoints/student", help="Checkpoint output dir")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=100_000)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=4_000)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pl.seed_everything(42)

    student = StudentTTS()
    speaker_encoder = SpeakerEncoder()
    teacher = None  # Loaded lazily inside DistillationTrainer to avoid OOM

    model = DistillationTrainer(
        student=student,
        speaker_encoder=speaker_encoder,
        teacher_name=args.teacher_model,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
    )

    print(f"[Student] Trainable params: {count_parameters(student):,}")
    print(f"[SpeakerEncoder] Trainable params: {count_parameters(speaker_encoder):,}")

    train_ds = TTSDataset(args.data_dir)
    # Simple 90/10 split
    train_size = int(0.9 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_set, val_set = torch.utils.data.random_split(train_ds, [train_size, val_size])

    # Windows shared-memory crash fix: single-process data loading
    train_loader = get_dataloader(train_set, args.batch_size, num_workers=0, shuffle=True, pin_memory=False, collate_fn=collate_fn)
    val_loader = get_dataloader(val_set, args.batch_size, num_workers=0, shuffle=False, pin_memory=False, collate_fn=collate_fn)

    logger = setup_logging("student_distill", save_dir=str(Path(args.output_dir).parent / "runs"))
    callbacks = get_checkpoint_callbacks(args.output_dir, every_n_hours=1)

    trainer = Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)

    # Export ONNX + quantize
    export_dir = Path(args.output_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    model.export_onnx(str(export_dir / "student.onnx"))
    model.quantize_int8(
        str(export_dir / "student.onnx"),
        str(export_dir / "student_int8.onnx"),
    )
    print(f"[Export] Saved to {export_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
