"""Synthetic training data generator.

Uses pretrained SpeechT5 as a teacher model to generate clean spectrogram/waveform
pairs, then adds synthetic corruption to create input-target pairs for Faraday
(mel enhancement) and Aether (waveform post-filter) training.
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

# SpeechT5 imports
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_teacher_models(device: str = "cuda"):
    """Load pretrained SpeechT5 TTS + HiFi-GAN vocoder."""
    print("[DataGen] Loading SpeechT5 teacher models...")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts", use_safetensors=True
    ).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(
        "microsoft/speecht5_hifigan", use_safetensors=True
    ).to(device)
    tts.eval()
    vocoder.eval()
    print("[DataGen] Teacher models loaded.")
    return processor, tts, vocoder


def synthesize_teacher(text: str, processor, tts, vocoder, speaker_emb, device: str):
    """Run SpeechT5 text → spectrogram → waveform."""
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        spectrogram = tts.generate_speech(inputs["input_ids"], speaker_emb)
        # spectrogram: [T, 80]
        wav = vocoder(spectrogram)  # [T_samples]

    return spectrogram.cpu(), wav.cpu()


def corrupt_mel_for_faraday(mel: torch.Tensor) -> torch.Tensor:
    """Add synthetic corruption to mel spectrogram."""
    # mel: [T, 80]
    corrupted = mel.clone()

    # 1. Gaussian noise (SNR ~15 dB)
    noise = torch.randn_like(corrupted) * corrupted.std() * 0.15
    corrupted = corrupted + noise

    # 2. Random frequency masking (0-15% of bins)
    num_bins = corrupted.shape[1]
    num_mask = random.randint(0, int(num_bins * 0.15))
    if num_mask > 0:
        mask_bins = random.sample(range(num_bins), num_mask)
        corrupted[:, mask_bins] = 0

    # 3. Random time masking (0-10% of frames)
    num_frames = corrupted.shape[0]
    num_tmask = random.randint(0, int(num_frames * 0.10))
    if num_tmask > 0:
        mask_frames = random.sample(range(num_frames), num_tmask)
        corrupted[mask_frames, :] = 0

    # 4. Mild blur (1D average pool along time, then interpolate back)
    if random.random() < 0.3:
        orig_len = corrupted.shape[0]
        blurred = F.avg_pool1d(
            corrupted.transpose(0, 1).unsqueeze(0), kernel_size=3, stride=1, padding=1
        )
        corrupted = blurred.squeeze(0).transpose(0, 1)

    return corrupted


def corrupt_waveform_for_aether(wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    """Add synthetic corruption to waveform."""
    corrupted = wav.clone()

    # 1. Gaussian noise (SNR ~20 dB)
    noise = torch.randn_like(corrupted) * corrupted.std() * 0.10
    corrupted = corrupted + noise

    # 2. Codec compression simulation: downsample + lowpass
    if random.random() < 0.4:
        target_sr = random.choice([8000, 12000])
        resampled = torchaudio.transforms.Resample(sr, target_sr)(corrupted)
        corrupted = torchaudio.transforms.Resample(target_sr, sr)(resampled)

    # 3. Mild clipping
    if random.random() < 0.3:
        threshold = corrupted.abs().max() * random.uniform(0.7, 0.95)
        corrupted = torch.clamp(corrupted, -threshold, threshold)

    return corrupted


def compute_mel_from_waveform(wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
    """Compute log-mel spectrogram for Aether conditioning."""
    mel_tf = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=1024, hop_length=256, n_mels=80
    )
    mel = mel_tf(wav.unsqueeze(0))
    mel = torch.log(mel + 1e-6)
    return mel[0]  # [80, T_mel]


def compute_f0_energy(mel: torch.Tensor) -> tuple:
    """Compute simple f0 and energy from mel."""
    energy = mel.mean(dim=0, keepdim=True)  # [1, T]
    # Placeholder f0: use energy as proxy (Aether is untrained, this is sufficient)
    f0 = energy.clone()
    return f0, energy


def split_text_into_chunks(text: str, max_chars: int = 200) -> list:
    """Split long text into sentence chunks."""
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) < max_chars:
            current += sent + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sent + ". "
    if current:
        chunks.append(current.strip())
    if not chunks:
        chunks = [text[:max_chars]]
    return chunks


def generate_pairs(
    text_chunks: list,
    processor,
    tts,
    vocoder,
    output_dir: str,
    num_pairs: int = 5000,
    device: str = "cuda",
):
    """Generate (input, target) pairs for Faraday and Aether."""
    out_path = Path(output_dir)
    faraday_dir = out_path / "faraday_pairs"
    aether_dir = out_path / "aether_pairs"
    faraday_dir.mkdir(parents=True, exist_ok=True)
    aether_dir.mkdir(parents=True, exist_ok=True)

    # Default speaker embedding (random but fixed for consistency)
    speaker_emb = torch.randn(1, 512).to(device)

    # If we need more pairs than chunks, loop over chunks repeatedly
    chunk_idx = 0
    pbar = tqdm(total=num_pairs, desc="Generating pairs")

    for pair_id in range(num_pairs):
        text = text_chunks[chunk_idx % len(text_chunks)]
        chunk_idx += 1

        try:
            spec, wav_16k = synthesize_teacher(text, processor, tts, vocoder, speaker_emb, device)
        except Exception as exc:
            print(f"[DataGen] Failed on chunk {chunk_id}: {exc}")
            continue

        # spec: [T, 80], wav_16k: [samples]
        # Resample waveform to 24kHz for Aether
        wav_24k = torchaudio.transforms.Resample(16000, 24000)(wav_16k)

        # Compute 24kHz mel for Aether conditioning
        mel_24k = compute_mel_from_waveform(wav_24k, sr=24000)  # [80, T_mel]

        # --- Faraday pair ---
        gt_mel = spec.T.unsqueeze(0)  # [1, 80, T]
        corrupted_mel = corrupt_mel_for_faraday(spec).T.unsqueeze(0)  # [1, 80, T]

        faraday_data = {
            "student_mel": corrupted_mel,
            "gt_mel": gt_mel,
            "text_emb": torch.randn(512),  # placeholder, will be replaced by real text emb
            "speaker_emb": speaker_emb.cpu()[0],
        }
        torch.save(faraday_data, faraday_dir / f"pair_{pair_id:06d}.pt")

        # --- Aether pair ---
        corrupted_wav = corrupt_waveform_for_aether(wav_24k, sr=24000)
        f0, energy = compute_f0_energy(mel_24k)

        aether_data = {
            "input_waveform": corrupted_wav.unsqueeze(0),  # [1, T]
            "target_waveform": wav_24k.unsqueeze(0),  # [1, T]
            "mel": mel_24k.unsqueeze(0),  # [1, 80, T_mel]
            "speaker_emb": speaker_emb.cpu()[0][:192],  # truncate to 192
            "f0": f0.unsqueeze(0),  # [1, T_mel]
            "energy": energy.unsqueeze(0),  # [1, T_mel]
        }
        torch.save(aether_data, aether_dir / f"pair_{pair_id:06d}.pt")

        pbar.update(1)

    pbar.close()
    print(f"[DataGen] Saved {num_pairs} pairs to {output_dir}")


def load_texts_from_book(book_dir: str) -> list:
    """Load and chunk text from parsed book JSON."""
    book_dir = Path(book_dir)
    chunks = []

    for json_file in sorted(book_dir.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        for chapter_data in data.values():
            text = chapter_data.get("text", "")
            if text:
                chunks.extend(split_text_into_chunks(text, max_chars=200))

    # Fallback if no parsed data
    if not chunks:
        chunks = [
            "Hello world. This is a test.",
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was silence.",
        ]

    print(f"[DataGen] Loaded {len(chunks)} text chunks from {book_dir}")
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--book_dir", default="./book_parsed")
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--num_pairs", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    processor, tts, vocoder = load_teacher_models(args.device)
    text_chunks = load_texts_from_book(args.book_dir)

    if len(text_chunks) == 0:
        raise ValueError("No text chunks found. Parse a book first.")

    generate_pairs(
        text_chunks=text_chunks,
        processor=processor,
        tts=tts,
        vocoder=vocoder,
        output_dir=args.output_dir,
        num_pairs=args.num_pairs,
        device=args.device,
    )


if __name__ == "__main__":
    main()
