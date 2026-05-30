"""Knowledge distillation LightningModule for student training."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .student import StudentTTS
from .speaker_encoder import SpeakerEncoder
from .vocoder import HiFiGenerator


class MelFeatureExtractor(nn.Module):
    """Simple CNN feature extractor for feature-matching loss."""

    def __init__(self, mel_bins: int = 80):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(mel_bins, 128, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2)),
            nn.Sequential(nn.Conv1d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2)),
            nn.Sequential(nn.Conv1d(256, 512, 3, padding=1), nn.ReLU()),
        ])

    def forward(self, mel: torch.Tensor) -> list:
        # mel: [B, mel_bins, T]
        feats = []
        x = mel
        for layer in self.layers:
            x = layer(x)
            feats.append(x)
        return feats


class DistillationTrainer(pl.LightningModule):
    """Train student with teacher distillation + ground-truth supervision."""

    def __init__(
        self,
        student: StudentTTS,
        speaker_encoder: SpeakerEncoder,
        teacher_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        lr: float = 2e-4,
        warmup_steps: int = 4000,
        teacher_weight: float = 0.5,
        gt_weight: float = 0.5,
        fm_weight: float = 0.1,
    ):
        super().__init__()
        self.student = student
        self.speaker_encoder = speaker_encoder
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.teacher_weight = teacher_weight
        self.gt_weight = gt_weight
        self.fm_weight = fm_weight

        self.feature_extractor = MelFeatureExtractor()
        self.teacher = None  # Lazy load to save VRAM
        self.teacher_name = teacher_name

    def setup(self, stage: Optional[str] = None):
        if self.teacher is None and stage == "fit":
            try:
                from TTS.api import TTS
                self.teacher = TTS(self.teacher_name).to(self.device)
                self.teacher.eval()
                for p in self.teacher.parameters():
                    p.requires_grad = False
            except Exception as e:
                print(f"[Warning] Could not load teacher model: {e}. Using GT-only loss.")
                self.teacher = None
                # Disable teacher distillation weight so loss computation is honest
                # and we don't waste compute on a teacher-less L1 term.
                self.teacher_weight = 0.0

    def forward(self, text_tokens, speaker_waveform):
        speaker_emb = self.speaker_encoder(speaker_waveform)
        mel = self.student(text_tokens, speaker_emb)
        return mel

    def training_step(self, batch, batch_idx):
        text_tokens = batch["text_tokens"]
        mel_gt = batch["mel"]
        teacher_mel = batch.get("teacher_mel", mel_gt)
        speaker_waveforms = batch["speaker_waveforms"]

        # Collate speaker waveforms to same length
        max_len = max(w.shape[-1] for w in speaker_waveforms)
        speaker_waveforms = torch.stack([
            F.pad(w, (0, max_len - w.shape[-1])) for w in speaker_waveforms
        ])

        speaker_emb = self.speaker_encoder(speaker_waveforms)
        mel_pred, _ = self.student(text_tokens, speaker_emb, mel_target=mel_gt)

        loss_gt = F.l1_loss(mel_pred, mel_gt)
        loss_teacher = F.l1_loss(mel_pred, teacher_mel)

        # Feature matching
        feat_pred = self.feature_extractor(mel_pred)
        feat_gt = self.feature_extractor(mel_gt)
        loss_fm = sum(F.l1_loss(a, b) for a, b in zip(feat_pred, feat_gt))

        loss = (
            self.gt_weight * loss_gt
            + self.teacher_weight * loss_teacher
            + self.fm_weight * loss_fm
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_l1_gt", loss_gt)
        self.log("train_l1_teacher", loss_teacher)
        self.log("train_fm", loss_fm)
        return loss

    def validation_step(self, batch, batch_idx):
        text_tokens = batch["text_tokens"]
        mel_gt = batch["mel"]
        speaker_waveforms = batch["speaker_waveforms"]
        max_len = max(w.shape[-1] for w in speaker_waveforms)
        speaker_waveforms = torch.stack([
            F.pad(w, (0, max_len - w.shape[-1])) for w in speaker_waveforms
        ])
        speaker_emb = self.speaker_encoder(speaker_waveforms)
        mel_pred, loss = self.student(text_tokens, speaker_emb, mel_target=mel_gt)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.98), weight_decay=0.01)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.warmup_steps, T_mult=2)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def export_onnx(self, path: str):
        dummy_tokens = torch.randint(0, 10_000, (1, 128), dtype=torch.long)
        dummy_spk = torch.randn(1, 192)
        torch.onnx.export(
            self.student,
            (dummy_tokens, dummy_spk),
            path,
            input_names=["text_tokens", "speaker_embedding"],
            output_names=["mel"],
            dynamic_axes={
                "text_tokens": {0: "batch", 1: "seq_len"},
                "speaker_embedding": {0: "batch"},
                "mel": {0: "batch", 2: "time"},
            },
            opset_version=17,
        )

    def quantize_int8(self, onnx_path: str, output_path: str):
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QInt8)
        except Exception as e:
            print(f"[Warning] INT8 quantization failed: {e}. Copying fp32 ONNX.")
            import shutil
            shutil.copy(onnx_path, output_path)
