"""Batch-convert all chapters in ./book to ./audiobook/."""

import time
import numpy as np
import soundfile as sf
from pathlib import Path

from pdf_parser import PDFParser
from demo_tts import DemonTTS


def main():
    out_dir = Path("./audiobook")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Convert] Loading models...")
    tts = DemonTTS()

    print("[Convert] Parsing PDFs...")
    parser = PDFParser()
    chapters = parser.parse_folder("./book")

    # Skip sample.pdf chapters, keep only real book
    book_chapters = {
        k: v for k, v in chapters.items()
        if "sample" not in k.lower()
    }

    print(f"[Convert] Found {len(book_chapters)} chapters to synthesize")

    voice = "Rick C-137"
    chapter_wavs = []
    start_all = time.time()

    for idx, (key, ch) in enumerate(book_chapters.items(), 1):
        text = ch["text"]
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in key)[:60]
        fpath = out_dir / f"{safe_name}.flac"

        print(f"\n[{idx}/{len(book_chapters)}] Synthesizing: {key[:50]}...")
        t0 = time.time()

        try:
            wav = tts.synthesize(text, voice_id=voice)
            tts.save_wav(wav, str(fpath))
            chapter_wavs.append(wav)
            elapsed = time.time() - t0
            print(f"  -> {len(wav)/24000:.1f}s audio in {elapsed:.1f}s ({len(text)} chars)")
        except Exception as exc:
            print(f"  -> FAILED: {exc}")

    # Combine with 0.5s silence
    if chapter_wavs:
        silence = np.zeros(int(0.5 * 24000), dtype=np.float32)
        combined = []
        for w in chapter_wavs:
            combined.append(w)
            combined.append(silence)
        combined.pop()
        full_wav = np.concatenate(combined)
        full_path = out_dir / "full_audiobook.flac"
        sf.write(str(full_path), full_wav, 24000)
        print(f"\n[Convert] Combined audiobook: {full_path} ({len(full_wav)/24000/60:.1f} min)")

    total = time.time() - start_all
    print(f"[Convert] Done in {total/60:.1f} minutes")


if __name__ == "__main__":
    main()
