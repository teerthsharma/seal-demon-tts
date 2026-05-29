"""Generate complete audiobook from parsed book JSON.

Processes each chapter through the full DemonTTS pipeline:
1. Parse chapter text
2. Chunk into segments
3. Synthesize each segment with Faraday + Aether + EPSILON-PHASE
4. Concatenate with proper pauses
5. Apply NeuralWhisper DSP post-processing
6. Export as FLAC chapters
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from demo_tts import DemonTTS
from dsp_postprocess import DSPPostProcessor


def load_book(parsed_path: str) -> dict:
    with open(parsed_path, "r", encoding="utf-8") as f:
        return json.load(f)


def synthesize_chapter(
    tts: DemonTTS,
    chapter_text: str,
    speaker_emb: np.ndarray,
    dsp: DSPPostProcessor,
) -> np.ndarray:
    """Synthesize one chapter with full pipeline."""
    chunks = tts._chunk_text(chapter_text)

    wav_segments = []
    for chunk in tqdm(chunks, desc="Synthesizing", leave=False):
        wav = tts.synthesize(
            chunk,
            speaker_emb=speaker_emb,
            use_faraday=True,
            use_aether=True,
        )
        wav_segments.append(wav)
        # Pause between chunks: 0.3s silence
        wav_segments.append(np.zeros(int(0.3 * 24000), dtype=np.float32))

    full_wav = np.concatenate(wav_segments)

    # DSP post-process
    wav_t = torch.from_numpy(full_wav)
    wav_t = dsp.process(wav_t)

    return wav_t.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--book",
        default="./book_parsed/Threshold's Pursuit_6b3bb078d03bc9c4.json",
    )
    parser.add_argument("--output_dir", default="./audiobook")
    parser.add_argument(
        "--voice_sample",
        default=None,
        help="Path to voice sample for cloning",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tts = DemonTTS()
    dsp = DSPPostProcessor(sample_rate=24000)

    if args.voice_sample:
        speaker_emb = tts.clone_voice(args.voice_sample)
        print(f"[Audiobook] Cloned voice from {args.voice_sample}")
    else:
        speaker_emb = np.zeros(192, dtype=np.float32)
        print("[Audiobook] Using default voice (no cloning)")

    book = load_book(args.book)

    for chapter_name, chapter_data in book.items():
        text = chapter_data.get("text", "")
        if not text.strip():
            continue

        print(f"\n[Audiobook] Chapter: {chapter_name}")
        wav = synthesize_chapter(tts, text, speaker_emb, dsp)

        safe_name = "".join(c if c.isalnum() else "_" for c in chapter_name)
        out_path = output_dir / f"{safe_name}.flac"
        sf.write(out_path, wav, 24000, format="FLAC")
        print(f"[Audiobook] Saved: {out_path} ({len(wav) / 24000:.1f}s)")

    print("\n🎧 Audiobook generation complete!")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
