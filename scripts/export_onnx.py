#!/usr/bin/env python3
"""Export DemonTTS models to ONNX for Rust pipeline inference.

Exports:
  - Faraday U-Net (noise prediction): [mel, t, speaker_emb] -> noise_pred
  - Aether FilterNet: [waveform, mel, speaker_emb, f0, energy] -> waveform
  - SpeakerEncoder: [waveform] -> speaker_emb (192-dim)
  - Vocoder (SpeechT5 HiFi-GAN): [mel] -> waveform

Usage:
    python scripts/export_onnx.py --output_dir ./checkpoints/export
"""

import argparse
import warnings
from pathlib import Path

import torch
import torchaudio

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from faraday.model import FaradayDiffusion
from aether.model import AetherFilter
from neural.speaker_encoder import SpeakerEncoder
from transformers import SpeechT5HifiGan

warnings.filterwarnings("ignore")


class FaradayOnnxWrapper(torch.nn.Module):
    """Wrapper that exports Faraday.forward with fixed text_emb=None."""
    def __init__(self, faraday: FaradayDiffusion):
        super().__init__()
        self.faraday = faraday

    def forward(self, mel: torch.Tensor, t: torch.Tensor, speaker_emb: torch.Tensor) -> torch.Tensor:
        return self.faraday(mel, t, text_emb=None, speaker_emb=speaker_emb)


class VocoderOnnxWrapper(torch.nn.Module):
    """Wrapper for SpeechT5 HiFi-GAN to accept [B, bins, T] and return [B, T_wav]."""
    def __init__(self, vocoder):
        super().__init__()
        self.vocoder = vocoder

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 80, T] -> vocoder expects [B, T, 80] internally
        # SpeechT5HifiGan.forward takes spectrogram of shape (batch_size, seq_len, model_dim)
        mel_t = mel.transpose(1, 2)  # [B, T, 80]
        wav = self.vocoder(mel_t)    # [B, T_wav]
        return wav                   # [B, T_wav]


def export_faraday(output_dir: Path, device: torch.device, mel_length: int = 256):
    print("[Export] Loading Faraday...")
    model = FaradayDiffusion(
        text_dim=512,
        speaker_dim=512,
        cond_dim=512,
        base_channels=192,
    ).to(device)
    model.eval()

    # Try to load trained weights
    ckpt = output_dir.parent / "faraday" / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"[Export] Loaded checkpoint: {ckpt}")
    else:
        print(f"[Export] No checkpoint found at {ckpt}, using initialized weights.")

    wrapper = FaradayOnnxWrapper(model).to(device).eval()
    dummy_mel = torch.randn(1, 1, 80, mel_length, device=device)
    dummy_t = torch.tensor([0], dtype=torch.long, device=device)
    dummy_spk = torch.randn(1, 512, device=device)

    out_path = output_dir / "faraday.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_mel, dummy_t, dummy_spk),
        str(out_path),
        input_names=["mel", "t", "speaker_emb"],
        output_names=["noise_pred"],
        dynamic_axes={
            "mel": {0: "batch", 3: "time"},
            "t": {0: "batch"},
            "speaker_emb": {0: "batch"},
            "noise_pred": {0: "batch", 3: "time"},
        },
        opset_version=14,
    )
    print(f"[Export] Faraday -> {out_path}")


def export_aether(output_dir: Path, device: torch.device):
    print("[Export] Loading Aether...")
    model = AetherFilter().to(device)
    model.eval()

    ckpt = output_dir.parent / "aether" / "best.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state.get("model_state_dict", state))
        print(f"[Export] Loaded checkpoint: {ckpt}")
    else:
        print(f"[Export] No checkpoint found at {ckpt}, using initialized weights.")

    out_path = output_dir / "aether.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.export_onnx(str(out_path))
    print(f"[Export] Aether -> {out_path}")


def export_speaker_encoder(output_dir: Path, device: torch.device):
    print("[Export] Loading SpeakerEncoder...")
    model = SpeakerEncoder().to(device)
    model.eval()

    ckpt = output_dir.parent / "speaker_encoder.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[Export] Loaded checkpoint: {ckpt}")
    else:
        print(f"[Export] No checkpoint found at {ckpt}, using initialized weights.")

    dummy_wav = torch.randn(1, 16000, device=device)
    out_path = output_dir / "speaker_encoder.onnx"
    torch.onnx.export(
        model,
        dummy_wav,
        str(out_path),
        input_names=["waveform"],
        output_names=["speaker_emb"],
        dynamic_axes={
            "waveform": {0: "batch", 1: "time"},
            "speaker_emb": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"[Export] SpeakerEncoder -> {out_path}")


def export_vocoder(output_dir: Path, device: torch.device, mel_length: int = 256):
    print("[Export] Loading SpeechT5 HiFi-GAN vocoder...")
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan", use_safetensors=True
    ).to(device)
    vocoder.eval()

    wrapper = VocoderOnnxWrapper(vocoder).to(device).eval()
    dummy_mel = torch.randn(1, 80, mel_length, device=device)
    out_path = output_dir / "vocoder.onnx"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            wrapper,
            dummy_mel,
            str(out_path),
            input_names=["mel"],
            output_names=["waveform"],
            dynamic_axes={
                "mel": {0: "batch", 2: "time"},
                "waveform": {0: "batch", 2: "time"},
            },
            opset_version=14,
        )
        print(f"[Export] Vocoder -> {out_path}")
    except Exception as e:
        print(f"[Export] Vocoder export FAILED (expected for some transformer ops): {e}")
        print("[Export] The Python pipeline will use the HuggingFace vocoder directly.")


def main():
    parser = argparse.ArgumentParser(description="Export DemonTTS models to ONNX")
    parser.add_argument("--output_dir", default="./checkpoints/export")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mel_length", type=int, default=256)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Export] Device: {device}")
    print(f"[Export] Output directory: {output_dir}")

    export_faraday(output_dir, device, args.mel_length)
    export_aether(output_dir, device)
    export_speaker_encoder(output_dir, device)
    export_vocoder(output_dir, device, args.mel_length)

    print("\n[Export] All exports complete.")
    print("Note: The Rust pipeline also requires a student ONNX model.")
    print("      Train the student model or use the Python pipeline (demo_tts.py) for now.")


if __name__ == "__main__":
    main()
