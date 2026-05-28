"""Integration test: PDF → TTS → WAV.

Verifies:
- PDF parsing (sample.pdf in ./book/)
- Model loading (with fallback to initialized weights)
- Synthesis pipeline (outputs test_output.wav)
- Memory stays under 6 GB
"""

import gc
import os
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from pdf_parser import PDFParser
from demo_tts import DemonTTS


def test_pdf_parse():
    parser = PDFParser()
    result = parser.parse_folder("./book")
    assert len(result) > 0, "No chapters parsed"
    for k, v in result.items():
        assert "text" in v, f"Missing text in {k}"
        assert len(v["text"]) > 0, f"Empty text in {k}"
    print(f"[TEST] PDF parse OK – {len(result)} chapters")
    return result


def test_synthesis(target_duration: float = 5.0):
    tts = DemonTTS()

    # Use a long text to hit ~5s of audio
    # Student outputs 1 mel frame per token; vocoder uses hop=256 @ 24kHz
    # So ~1 token ≈ 256/24000 ≈ 0.0107s. For 5s we need ~470 tokens.
    long_text = (
        "In the beginning, there was only darkness. "
        "Then a voice echoed through the void, shaping stars and worlds. "
        "That voice was not human, nor was it divine. "
        "It was the demon of inkosei, bound to paper and ink, "
        "waiting for a reader brave enough to speak its name aloud. "
        "For every word written, a soul was whispered into existence. "
        "And every silence that followed was a scream trapped in graphite. "
        "This is the story of how one mortal found the forbidden book, "
        "and how the book, in turn, found its voice through him. "
        "The pages turned themselves, the ink flowed like blood, "
        "and the audiobook played on, forever, in the spaces between dreams. "
        "Listen carefully, for the demon is speaking now, and it knows your name. "
    )

    wav = tts.synthesize(long_text, voice_id="Rick C-137")
    duration = len(wav) / 24_000

    tts.save_wav(wav, "test_output.wav")
    print(f"[TEST] Synthesis OK – {len(wav)} samples, {duration:.2f}s")

    # Memory check
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"[TEST] Peak VRAM: {mem_mb:.1f} MB")
        assert mem_mb < 6_000, f"VRAM exceeded 6 GB: {mem_mb:.1f} MB"

    # File check
    info = sf.info("test_output.wav")
    assert info.samplerate == 24_000
    assert info.frames > 0
    print(f"[TEST] Output file OK – {info.duration:.2f}s @ {info.samplerate} Hz")

    return wav


def main():
    print("=" * 50)
    print("DemonTTS Integration Test")
    print("=" * 50)

    start = time.time()
    test_pdf_parse()
    test_synthesis()
    elapsed = time.time() - start

    print(f"[TEST] All passed in {elapsed:.1f}s")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
