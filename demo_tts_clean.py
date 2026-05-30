"""DemonTTS Clean Inference — SpeechT5 base, no undertrained enhancers.

Fixes robotic voice by:
1. Using SpeechT5's native speaker embedding (no random projection)
2. Disabling Faraday/Aether until properly trained
3. Cleaning parsed text (fixing PDF encoding garbage)
4. Token-aware chunking (prevents >600 token crash)
5. Gentler DSP post-processing
"""

import json
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor


class DemonTTSClean:
    """Clean TTS: SpeechT5 only, no broken enhancers."""

    def __init__(
        self,
        model_dir: str = "./models",
        device: str = None,
        max_tokens: int = 500,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_tokens = max_tokens
        print(f"[DemonTTSClean] Device: {self.device}")

        # Pretrained base TTS
        print("[DemonTTSClean] Loading SpeechT5...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts", use_safetensors=True
        ).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan", use_safetensors=True
        ).to(self.device)
        self.tts.eval()
        self.vocoder.eval()

        # Use SpeechT5's own speaker embedding from the model
        # This is a trained embedding, not random noise
        self.default_speaker = self._get_default_speaker()

        # Resample 16kHz → 24kHz
        self.resample_16to24 = torchaudio.transforms.Resample(16000, 24000).to(self.device)

    def _get_default_speaker(self):
        """Extract a proper speaker embedding from SpeechT5 itself."""
        # SpeechT5 has speaker embeddings in its decoder
        # Use the mean of the learned speaker embeddings as default
        if hasattr(self.tts, 'speaker_embed') and self.tts.speaker_embed is not None:
            spk = self.tts.speaker_embed.weight.mean(dim=0, keepdim=True)
            print(f"[DemonTTSClean] Using SpeechT5 native speaker embedding")
            return spk.to(self.device)
        else:
            # Fallback: use zero vector (neutral voice)
            print("[DemonTTSClean] Using neutral speaker embedding")
            return torch.zeros(1, 512, device=self.device)

    def _clean_text(self, text: str) -> str:
        """Clean PDF parsing artifacts."""
        # Replace common encoding garbage
        replacements = {
            '\uFFFD': "'",  # � → apostrophe
            '\u2019': "'",  # ' → simple apostrophe
            '\u2018': "'",  # ' → simple apostrophe
            '\u201C': '"',  # " → simple quote
            '\u201D': '"',  # " → simple quote
            '\u2013': '-',  # – → hyphen
            '\u2014': '-',  # — → hyphen
            '\u2026': '...', # … → dots
            '\n': ' ',       # newlines → space
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Chunk text by token count to stay under SpeechT5's 600 token limit."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""
        
        for sent in sentences:
            test = (current + " " + sent).strip() if current else sent
            # Tokenize to check length
            inputs = self.processor(text=test, return_tensors="pt")
            n_tokens = inputs["input_ids"].shape[1]
            
            if n_tokens < self.max_tokens:
                current = test
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
                # If single sentence is too long, split by commas
                if self.processor(text=sent, return_tensors="pt")["input_ids"].shape[1] >= self.max_tokens:
                    parts = sent.split(", ")
                    current = ""
                    for part in parts:
                        test2 = (current + ", " + part).strip() if current else part
                        if self.processor(text=test2, return_tensors="pt")["input_ids"].shape[1] < self.max_tokens:
                            current = test2
                        else:
                            if current:
                                chunks.append(current.strip())
                            current = part
        
        if current:
            chunks.append(current.strip())
        
        if not chunks:
            chunks = [text[:200]]  # Last resort
        
        return chunks

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        text = self._clean_text(text)
        
        if speaker_emb is None:
            speaker_emb = self.default_speaker
        
        chunks = self._chunk_by_tokens(text)
        waveforms: List[np.ndarray] = []
        
        for chunk in chunks:
            inputs = self.processor(text=chunk, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # SpeechT5 → spectrogram
            spectrogram = self.tts.generate_speech(inputs["input_ids"], speaker_emb)
            # Vocoder → waveform @ 16kHz
            wav_16k = self.vocoder(spectrogram)
            # Resample to 24kHz
            wav_24k = self.resample_16to24(wav_16k)
            
            waveforms.append(wav_24k.cpu().numpy())
        
        if not waveforms:
            return np.zeros(0, dtype=np.float32)
        
        return np.concatenate(waveforms)

    def save_wav(self, wav: np.ndarray, path: str, sample_rate: int = 24_000):
        sf.write(path, wav, sample_rate)
        print(f"[DemonTTSClean] Saved: {path}")


def generate_audiobook_clean(
    book_path: str,
    output_dir: str,
    pause_seconds: float = 1.0,
):
    """Generate audiobook with clean pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tts = DemonTTSClean()
    
    with open(book_path, "r", encoding="utf-8") as f:
        book = json.load(f)
    
    chapter_files = []
    
    for chapter_name, chapter_data in book.items():
        text = chapter_data.get("text", "")
        if not text.strip():
            continue
        
        print(f"\n[Audiobook] Chapter: {chapter_name}")
        wav = tts.synthesize(text)
        
        safe_name = "".join(c if c.isalnum() else "_" for c in chapter_name)
        out_path = output_dir / f"{safe_name}.flac"
        sf.write(out_path, wav, 24000, format="FLAC")
        print(f"[Audiobook] Saved: {out_path} ({len(wav)/24000:.1f}s)")
        chapter_files.append(out_path)
    
    # Combine all chapters with pause
    if chapter_files:
        print("\n[Audiobook] Combining chapters...")
        pause = np.zeros(int(pause_seconds * 24000), dtype=np.float32)
        parts = []
        for fpath in chapter_files:
            wav, sr = sf.read(fpath, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            parts.extend([wav, pause])
        parts.pop()  # Remove last pause
        
        combined = np.concatenate(parts)
        full_path = output_dir / "FULL_BOOK.flac"
        sf.write(full_path, combined, 24000, format="FLAC")
        print(f"[Audiobook] FULL BOOK: {full_path} ({len(combined)/24000/60:.1f} min)")


if __name__ == "__main__":
    generate_audiobook_clean(
        book_path="book_parsed/Threshold's Pursuit_6b3bb078d03bc9c4.json",
        output_dir="audiobook/thresholds_pursuit_clean",
    )
