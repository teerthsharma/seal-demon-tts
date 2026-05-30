#!/usr/bin/env python3
"""Convert existing Faraday/Aether pairs into Student-compatible training data.

Uses the trained BPE tokenizer to encode text, and reuses existing mel/audio.
No re-synthesis needed — maps pair IDs back to their source text chunks.
"""

import json
import sys
from pathlib import Path

import torch
import torchaudio
from tokenizers import Tokenizer
from tqdm import tqdm


def load_text_chunks(book_dir: str) -> list:
    chunks = []
    for json_file in sorted(Path(book_dir).glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        for chapter_data in data.values():
            text = chapter_data.get("text", "")
            if text:
                chunks.append(text)
    if not chunks:
        raise ValueError(f"No text found in {book_dir}")
    return chunks


def main():
    data_dir = Path("./data")
    faraday_dir = data_dir / "faraday_pairs"
    aether_dir = data_dir / "aether_pairs"
    student_dir = data_dir / "student_pairs"
    student_dir.mkdir(parents=True, exist_ok=True)

    if not faraday_dir.exists() or not aether_dir.exists():
        print("[ERROR] Faraday or Aether pairs not found. Run generate_training_data.py first.")
        sys.exit(1)

    # Load tokenizer
    tokenizer = Tokenizer.from_file("models/tokenizer.json")
    print(f"[Convert] Tokenizer vocab: {len(tokenizer.get_vocab())}")

    # Load text chunks
    text_chunks = load_text_chunks("book_parsed")
    print(f"[Convert] Loaded {len(text_chunks)} text chunks")

    # Find existing pair IDs
    faraday_files = sorted(faraday_dir.glob("pair_*.pt"))
    print(f"[Convert] Found {len(faraday_files)} Faraday pairs to convert")

    # Resampler: Aether waveforms are 24kHz, student speaker encoder expects 16kHz
    resample_24to16 = torchaudio.transforms.Resample(24000, 16000)

    for faraday_path in tqdm(faraday_files, desc="Converting to student pairs"):
        pair_id = int(faraday_path.stem.split("_")[1])
        aether_path = aether_dir / faraday_path.name
        out_path = student_dir / faraday_path.name

        if out_path.exists():
            continue

        if not aether_path.exists():
            tqdm.write(f"[WARN] Missing Aether pair for {pair_id}, skipping")
            continue

        # Map pair_id -> text chunk
        text = text_chunks[pair_id % len(text_chunks)]
        encoded = tokenizer.encode(text)
        text_tokens = torch.tensor(encoded.ids, dtype=torch.long)

        # Load Faraday mel (ground truth)
        faraday_data = torch.load(faraday_path, weights_only=True)
        mel = faraday_data["gt_mel"]  # [1, 80, T]
        if mel.dim() == 3:
            mel = mel.squeeze(0)  # [80, T]

        # Load Aether waveform and resample to 16kHz
        aether_data = torch.load(aether_path, weights_only=True)
        wav_24k = aether_data["target_waveform"]  # [1, T]
        if wav_24k.dim() == 2:
            wav_24k = wav_24k.squeeze(0)  # [T]
        wav_16k = resample_24to16(wav_24k)

        student_data = {
            "text_tokens": text_tokens,
            "mel": mel,
            "speaker_waveform": wav_16k,
            "teacher_mel": mel,  # Fallback: self-distillation from GT
        }
        torch.save(student_data, out_path)

    print(f"[Convert] Done. Student pairs saved to {student_dir}")


if __name__ == "__main__":
    main()
