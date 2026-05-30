#!/usr/bin/env python3
"""Judged Audiobook Generator — Chapter-by-chapter with self-judge + Faraday injection.

Pipeline per chapter:
  1. Draft synthesis (SpeechT5 → Faraday → Vocoder → Aether)
  2. Self-judge (clipping, silence, dynamic range, spectral variance)
  3. Faraday injection judge (re-diffuse with perturbed speaker embedding, pick best)
  4. DSP polish → save to audiobook/final_7hr/
  5. Emit student training pair from polished output so student learns from council quality

Student training is decoupled — this script runs even if student crashed earlier.
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from tokenizers import Tokenizer

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent))

from demo_tts import DemonTTS
from dsp_postprocess import DSPPostProcessor


class ChapterJudge:
    """Heuristic quality judge for synthesized audio.

    Scores clipping, silence ratio, dynamic range, and spectral variance.
    Higher score = better.  Target > 0.7 for acceptance.
    """

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=80
        )

    def score(self, wav: np.ndarray) -> float:
        x = torch.from_numpy(wav.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # --- temporal metrics ---
        abs_x = x.abs()
        clipping_ratio = (abs_x > 0.99).float().mean().item()
        silence_ratio = (abs_x < 0.005).float().mean().item()
        dynamic_range = (abs_x.quantile(0.95) - abs_x.quantile(0.05)).item()

        # --- spectral metrics ---
        mel = self.mel_transform(x)  # [1, 80, T]
        mel_db = torch.log(mel + 1e-6)
        # spectral centroid variance across time
        freqs = torch.linspace(0, self.sample_rate // 2, 80).unsqueeze(1)  # [80, 1]
        energy_per_frame = mel.sum(dim=1, keepdim=True) + 1e-8  # [1, 1, T]
        centroid = (freqs.unsqueeze(0) * mel).sum(dim=1) / energy_per_frame.squeeze(1)  # [1, T]
        spectral_std = centroid.std().item()

        # --- composite score ---
        # penalties
        clip_penalty = min(clipping_ratio * 50.0, 0.5)
        silence_penalty = max(0.0, (silence_ratio - 0.25) * 1.0)
        # bonuses
        dr_bonus = min(dynamic_range * 2.0, 0.3)
        spec_bonus = min(spectral_std / 500.0, 0.3)

        score = 1.0 - clip_penalty - silence_penalty + dr_bonus + spec_bonus
        return float(np.clip(score, 0.0, 1.0))

    def diagnose(self, wav: np.ndarray) -> dict:
        x = torch.from_numpy(wav.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        abs_x = x.abs()
        return {
            "clipping_ratio": float((abs_x > 0.99).float().mean().item()),
            "silence_ratio": float((abs_x < 0.005).float().mean().item()),
            "dynamic_range": float((abs_x.quantile(0.95) - abs_x.quantile(0.05)).item()),
            "peak": float(abs_x.max().item()),
            "rms": float(x.pow(2).mean().sqrt().item()),
        }


class FaradayInjectionJudge:
    """Re-run Faraday with perturbed speaker embeddings (injection) and pick best.

    Since the learned Arbiter is not yet trained, we use the heuristic ChapterJudge
    to score each variant.  Once FaradayArbiter is trained, swap in arbiter scores.
    """

    def __init__(self, tts: DemonTTS, judge: ChapterJudge, num_variants: int = 2):
        self.tts = tts
        self.judge = judge
        self.num_variants = num_variants

    def pick_best(self, text: str, base_speaker_emb: np.ndarray) -> tuple:
        """Generate N variants with injected speaker noise, return best waveform + score."""
        variants = []
        base = torch.from_numpy(base_speaker_emb).float().unsqueeze(0).to(self.tts.device)

        for i in range(self.num_variants):
            # injection: small speaker perturbation
            if i == 0:
                spk = base
            else:
                torch.manual_seed(42 + i)
                noise = torch.randn_like(base) * 0.03 * i  # increasing perturbation
                spk = base + noise

            spk_np = spk.squeeze(0).cpu().numpy()
            try:
                wav = self.tts.synthesize(text, speaker_emb=spk_np, use_faraday=True, use_aether=True)
                sc = self.judge.score(wav)
                variants.append((wav, sc, spk_np, i))
            except Exception as e:
                print(f"    [InjectionJudge] Variant {i} failed: {e}")
                continue

        if not variants:
            # fallback: synthesize without any fancy conditioning
            wav = self.tts.synthesize(text, speaker_emb=base_speaker_emb, use_faraday=True, use_aether=True)
            return wav, 0.5, base_speaker_emb, {"variants": 0, "best_idx": -1, "score": 0.5}

        # pick highest score
        best = max(variants, key=lambda x: x[1])
        metadata = {
            "variants": len(variants),
            "best_idx": best[3],
            "score": best[1],
            "all_scores": [v[1] for v in variants],
        }
        return best[0], best[1], best[2], metadata


def save_student_pair(text: str, wav: np.ndarray, speaker_emb: np.ndarray, tokenizer, out_path: Path):
    """Save a polished chapter as a student training pair."""
    # Tokenize text
    encoded = tokenizer.encode(text)
    text_tokens = torch.tensor(encoded.ids, dtype=torch.long)

    # Mel spectrogram @ 24kHz
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80
    )
    wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
    mel = mel_transform(wav_t)  # [1, 80, T]
    mel = torch.log(mel + 1e-6).squeeze(0)  # [80, T]

    # Speaker waveform @ 16kHz (student speaker encoder expects 16kHz)
    if len(wav) > 0:
        wav_t_24k = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        wav_16k = torchaudio.transforms.Resample(24000, 16000)(wav_t_24k).squeeze(0)
    else:
        wav_16k = torch.zeros(16000 * 3, dtype=torch.float32)

    pair = {
        "text_tokens": text_tokens,
        "mel": mel,
        "speaker_waveform": wav_16k,
        "teacher_mel": mel,  # self-supervise from polished council output
    }
    torch.save(pair, out_path)


def main():
    parser = argparse.ArgumentParser(description="Generate judged audiobook chapter-by-chapter")
    parser.add_argument("--book_dir", default="./book_parsed", help="Directory with parsed JSON books")
    parser.add_argument("--output_dir", default="./audiobook/final_7hr", help="Audiobook output directory")
    parser.add_argument("--student_data_dir", default="./data/student_pairs_from_audiobook", help="Where to emit student training pairs")
    parser.add_argument("--voice_sample", default=None, help="Path to voice sample for cloning")
    parser.add_argument("--min_score", type=float, default=0.5, help="Minimum self-judge score to accept a chapter")
    parser.add_argument("--max_retries", type=int, default=2, help="How many times to re-synthesize if score is too low")
    parser.add_argument("--device", default="cuda", help="Torch device")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student_dir = Path(args.student_data_dir)
    student_dir.mkdir(parents=True, exist_ok=True)

    # --- Init models ---
    print("[JudgePipeline] Loading DemonTTS...")
    tts = DemonTTS(device=args.device)
    dsp = DSPPostProcessor(sample_rate=24000)
    judge = ChapterJudge(sample_rate=24000)
    injection = FaradayInjectionJudge(tts, judge, num_variants=2)

    # Load tokenizer for student pair export
    tokenizer_path = Path("models/tokenizer.json")
    tokenizer = None
    if tokenizer_path.exists():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        print(f"[JudgePipeline] Tokenizer loaded: {len(tokenizer.get_vocab())} tokens")
    else:
        print("[WARN] Tokenizer not found — student pairs will NOT be emitted")

    # Voice
    if args.voice_sample:
        speaker_emb = tts.clone_voice(args.voice_sample)
        print(f"[JudgePipeline] Cloned voice from {args.voice_sample}")
    else:
        speaker_emb = np.zeros(192, dtype=np.float32)
        print("[JudgePipeline] Using default voice")

    # Find parsed books
    book_files = sorted(Path(args.book_dir).glob("*.json"))
    if not book_files:
        print(f"[ERROR] No parsed books in {args.book_dir}")
        sys.exit(1)

    print(f"[JudgePipeline] Found {len(book_files)} book(s)")

    pause = np.zeros(int(1.0 * 24000), dtype=np.float32)
    all_parts = []
    chapter_idx = 0
    student_pair_idx = 0

    for book_file in book_files:
        print(f"\n{'='*60}")
        print(f"[Book] {book_file.name}")
        print(f"{'='*60}")
        with open(book_file, "r", encoding="utf-8") as f:
            book = json.load(f)

        for chapter_name, chapter_data in book.items():
            text = chapter_data.get("text", "")
            if not text.strip():
                continue

            chapter_idx += 1
            safe_name = "".join(c if c.isalnum() else "_" for c in chapter_name)
            print(f"\n[{chapter_idx}] {chapter_name} — {len(text)} chars")

            # --- Self-judge + Faraday injection loop ---
            best_wav = None
            best_score = -1.0
            best_meta = {}

            for attempt in range(args.max_retries + 1):
                print(f"  Attempt {attempt + 1}/{args.max_retries + 1}...")
                wav, score, used_spk, meta = injection.pick_best(text, speaker_emb)
                diag = judge.diagnose(wav)
                print(f"    Score={score:.3f}  peak={diag['peak']:.3f}  clip={diag['clipping_ratio']:.4f}  "
                      f"silence={diag['silence_ratio']:.3f}  DR={diag['dynamic_range']:.3f}  "
                      f"best_variant={meta['best_idx']}  scores={meta['all_scores']}")

                if score > best_score:
                    best_score = score
                    best_wav = wav
                    best_meta = meta

                if score >= args.min_score:
                    print(f"  ✓ ACCEPTED on attempt {attempt + 1}")
                    break
                elif attempt < args.max_retries:
                    print(f"  ↻ RETRYING (score {score:.3f} < {args.min_score})")
                else:
                    print(f"  ⚠ USING BEST (score {best_score:.3f} after {args.max_retries + 1} attempts)")

            # --- DSP polish ---
            wav_t = torch.from_numpy(best_wav.astype(np.float32))
            wav_polished = dsp.process(wav_t).numpy()

            # --- Save chapter ---
            ch_path = out_dir / f"{chapter_idx:03d}_{safe_name}.flac"
            sf.write(ch_path, wav_polished, 24000, format="FLAC")
            print(f"  → Saved: {ch_path}  ({len(wav_polished)/24000:.1f}s)")

            all_parts.extend([wav_polished, pause])

            # --- Emit student pair from polished output ---
            if tokenizer is not None:
                pair_path = student_dir / f"pair_{student_pair_idx:06d}.pt"
                try:
                    save_student_pair(text, wav_polished, speaker_emb, tokenizer, pair_path)
                    print(f"  → Student pair: {pair_path}")
                    student_pair_idx += 1
                except Exception as e:
                    print(f"  → Student pair FAILED: {e}")

    # --- Full book concatenation ---
    if len(all_parts) > 1:
        all_parts.pop()  # remove trailing pause
    if all_parts:
        full = np.concatenate(all_parts)
        full_path = out_dir / "FULL_Audiobook_Male_Voice.flac"
        sf.write(full_path, full, 24000, format="FLAC")
        print(f"\n{'='*60}")
        print(f"FULL BOOK: {full_path}")
        print(f"Duration: {len(full)/24000/60:.1f} minutes")
        print(f"Chapters: {chapter_idx}")
        print(f"Student pairs emitted: {student_pair_idx}")
        print(f"{'='*60}")

    # Prevent Windows CUDA cleanup crash from killing parent shell
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
