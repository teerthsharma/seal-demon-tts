"""DemonTTS Chapter 2 Master Pipeline — fp16 inference, full stack.

Fits in 8GB VRAM by running all models in fp16/mixed precision.
Pipeline: SpeechT5 → Faraday(fp16) → HiFi-GAN(fp16) → Aether(fp16) → DSP

Master chapter: "2. Bound By Will" — train everything to match this chapter's style.
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

# Import enhancers
from aether.model import AetherFilter
from faraday.model import FaradayDiffusion
from dsp_postprocess import DSPPostProcessor

warnings.filterwarnings("ignore")


class DemonTTSMaster:
    """Full pipeline with fp16 memory optimization for 8GB cards."""

    def __init__(
        self,
        models_dir: str = "./models",
        device: str = None,
        max_tokens: int = 500,
        use_fp16: bool = True,
    ):
        self.models_dir = Path(models_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_tokens = max_tokens
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.dtype = torch.float16 if self.use_fp16 else torch.float32
        
        print(f"[Master] Device: {self.device}, dtype: {self.dtype}")

        # SpeechT5 (keep in fp32, it's small enough)
        print("[Master] Loading SpeechT5...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts", use_safetensors=True
        ).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan", use_safetensors=True
        ).to(self.device)
        self.tts.eval()
        self.vocoder.eval()

        # Native speaker embedding from SpeechT5 itself
        self.default_speaker = self._get_native_speaker()
        
        # Faraday (load in fp16)
        print("[Master] Loading Faraday...")
        self.faraday = FaradayDiffusion(
            text_dim=512, speaker_dim=512, cond_dim=512, base_channels=192
        ).to(self.device).to(self.dtype)
        faraday_path = self.models_dir / "faraday.pt"
        if faraday_path.exists():
            ckpt = torch.load(faraday_path, map_location=self.device, weights_only=True)
            self.faraday.load_state_dict(ckpt)
            print(f"[Master] Faraday loaded: {sum(p.numel() for p in self.faraday.parameters()):,} params")
        else:
            print("[Master] WARNING: Faraday checkpoint not found")
        self.faraday.eval()
        
        # Aether (load in fp16)
        print("[Master] Loading Aether...")
        self.aether = AetherFilter().to(self.device).to(self.dtype)
        aether_path = self.models_dir / "aether.pt"
        if aether_path.exists():
            ckpt = torch.load(aether_path, map_location=self.device, weights_only=False)
            self.aether.load_state_dict(ckpt)
            print(f"[Master] Aether loaded: {sum(p.numel() for p in self.aether.parameters()):,} params")
        else:
            print("[Master] WARNING: Aether checkpoint not found")
        self.aether.eval()
        
        # DSP
        self.dsp = DSPPostProcessor(sample_rate=24000)
        
        # Resample
        self.resample_16to24 = torchaudio.transforms.Resample(16000, 24000).to(self.device).to(self.dtype)
        
        # Mel transform for Aether (must match pipeline dtype to avoid Float/Half matmul errors)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80
        ).to(self.device).to(self.dtype)
        
        # Report VRAM
        if self.device.type == "cuda":
            used = torch.cuda.memory_allocated() / 1e9
            print(f"[Master] VRAM after load: {used:.2f} GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    def _get_native_speaker(self):
        """Extract trained speaker embedding from SpeechT5 decoder."""
        if hasattr(self.tts, 'speaker_embed') and self.tts.speaker_embed is not None:
            spk = self.tts.speaker_embed.weight.mean(dim=0, keepdim=True)
            print("[Master] Using SpeechT5 native speaker embedding")
            return spk.to(self.device)
        print("[Master] Using neutral speaker")
        return torch.zeros(1, 512, device=self.device)

    def _clean_text(self, text: str) -> str:
        """Remove PDF encoding garbage."""
        replacements = {
            '\uFFFD': "'", '\u2019': "'", '\u2018': "'",
            '\u201C': '"', '\u201D': '"', '\u2013': '-',
            '\u2014': '-', '\u2026': '...', '\n': ' ',
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _chunk_by_tokens(self, text: str) -> List[str]:
        """Chunk by token count to stay under SpeechT5's 600 limit."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""
        
        for sent in sentences:
            test = (current + " " + sent).strip() if current else sent
            inputs = self.processor(text=test, return_tensors="pt")
            n_tokens = inputs["input_ids"].shape[1]
            
            if n_tokens < self.max_tokens:
                current = test
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
                # Emergency split for long sentences
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
            chunks = [text[:200]]
        return chunks

    @torch.inference_mode()
    def synthesize(self, text: str) -> np.ndarray:
        text = self._clean_text(text)
        chunks = self._chunk_by_tokens(text)
        waveforms = []
        
        for chunk in chunks:
            inputs = self.processor(text=chunk, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Stage 1: SpeechT5 → spectrogram
            spectrogram = self.tts.generate_speech(inputs["input_ids"], self.default_speaker)
            mel = spectrogram.T.unsqueeze(0)  # [1, 80, T]
            
            # Stage 2: Faraday enhancement (supervised mode, fp16)
            mel_fp16 = mel.unsqueeze(0).to(self.dtype)  # [1, 1, 80, T]
            text_emb = torch.zeros(1, 512, device=self.device, dtype=self.dtype)
            spk_fp16 = self.default_speaker.to(self.dtype)
            mel_enhanced = self.faraday.supervised_enhance(
                mel_fp16, text_emb=text_emb, speaker_emb=spk_fp16
            )
            mel = mel_enhanced.squeeze(1).float()  # Back to fp32 for vocoder
            
            # Stage 3: Vocoder → waveform @ 16kHz
            wav_16k = self.vocoder(mel.squeeze(0).T)
            
            # Stage 4: Resample to 24kHz
            wav_24k = self.resample_16to24(wav_16k)
            
            # Stage 5: Aether waveform polish (fp16)
            wav_t = wav_24k.unsqueeze(0).unsqueeze(0).to(self.dtype)
            aether_mel = self.mel_transform(wav_t.squeeze(1))
            aether_mel = torch.log(aether_mel + 1e-6).to(self.dtype)
            T_mel = aether_mel.shape[2]
            energy = aether_mel.mean(dim=1, keepdim=True)
            f0 = torch.zeros(1, 1, T_mel, device=self.device, dtype=self.dtype)
            spk_aether = torch.randn(1, 192, device=self.device, dtype=self.dtype)
            wav_refined = self.aether(wav_t, aether_mel, spk_aether, f0, energy)
            wav_24k = wav_refined[0, 0].float()
            
            waveforms.append(wav_24k.cpu().numpy())
        
        if not waveforms:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(waveforms)

    def save(self, wav: np.ndarray, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, wav, 24000)
        print(f"[Master] Saved: {path}")


def generate_master_chapter():
    """Generate Chapter 2 (Bound By Will) as the master reference."""
    print("=" * 60)
    print("CHAPTER 2 MASTER — Generating reference audio")
    print("=" * 60)
    
    tts = DemonTTSMaster(use_fp16=True)
    
    with open("book_parsed/Threshold's Pursuit_6b3bb078d03bc9c4.json", "r", encoding="utf-8") as f:
        book = json.load(f)
    
    # Get Chapter 2
    chapter_key = "2. Bound By Will"
    chapter_data = book.get(chapter_key)
    if not chapter_data:
        print(f"ERROR: {chapter_key} not found in book")
        return
    
    text = chapter_data.get("text", "")
    print(f"\n[Master] Chapter: {chapter_key}")
    print(f"[Master] Text length: {len(text)} chars")
    
    wav = tts.synthesize(text)
    
    out_dir = Path("audiobook/master_chapter2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "02_Bound_By_Will_MASTER.flac"
    sf.write(out_path, wav, 24000, format="FLAC")
    print(f"\n[Master] MASTER CHAPTER GENERATED")
    print(f"[Master] Path: {out_path}")
    print(f"[Master] Duration: {len(wav)/24_000:.1f}s")
    print(f"[Master] File size: {out_path.stat().st_size / 1e6:.1f} MB")
    
    return out_path


if __name__ == "__main__":
    generate_master_chapter()
