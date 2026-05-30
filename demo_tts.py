"""DemonTTS inference engine — SpeechT5 base + Faraday + Aether."""

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

from aether.model import AetherFilter, count_parameters as aether_count
from faraday.model import FaradayDiffusion, count_parameters as faraday_count
from neural.speaker_encoder import SpeakerEncoder, count_parameters as spk_count


class DemonTTS:
    """End-to-end TTS: SpeechT5 → Faraday → Vocoder → Aether."""

    def __init__(
        self,
        model_dir: str = "./models",
        device: str = None,
        max_chars: int = 400,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_chars = max_chars
        print(f"[DemonTTS] Device: {self.device}")

        # Pretrained base TTS
        print("[DemonTTS] Loading SpeechT5 teacher...")
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.tts = SpeechT5ForTextToSpeech.from_pretrained(
            "microsoft/speecht5_tts", use_safetensors=True
        ).to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            "microsoft/speecht5_hifigan", use_safetensors=True
        ).to(self.device)
        self.tts.eval()
        self.vocoder.eval()

        # Enhancement networks
        self.faraday = self._load_model("faraday.pt", FaradayDiffusion)
        self.aether = self._load_model("aether.pt", AetherFilter)
        self.speaker_encoder = self._load_model("speaker_encoder.pt", SpeakerEncoder)

        # Default speaker embedding for SpeechT5 (512-dim) — learned from encoder if available
        self.default_speaker = torch.randn(1, 512).to(self.device)

        # Persistent projection layers — created ONCE, not in inference loop
        # Faraday expects 512-dim speaker (matches SpeechT5 training data)
        self.spk_proj_faraday = torch.nn.Linear(192, 512).to(self.device)
        # Aether expects 192-dim speaker
        self.spk_proj_aether = torch.nn.Linear(512, 192).to(self.device)

        # Persistent resample — created ONCE
        self.resample_16to24 = torchaudio.transforms.Resample(16000, 24000).to(self.device)

        # Mel transform for Aether conditioning (24kHz)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80
        ).to(self.device)

        self.voices_path = self.model_dir / "voices.json"
        self.voices: Dict[str, np.ndarray] = self._load_voices()

    def _load_model(self, filename: str, cls):
        path = self.model_dir / filename
        if cls is FaradayDiffusion:
            model = cls(
                text_dim=512,
                speaker_dim=512,
                cond_dim=512,
                base_channels=192,
            ).to(self.device)
        else:
            model = cls().to(self.device)
        if path.exists():
            try:
                state = torch.load(path, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                print(f"[DemonTTS] Loaded checkpoint: {filename}")
            except Exception as exc:
                warnings.warn(
                    f"Failed to load {filename}: {exc}. Using initialized weights."
                )
        else:
            warnings.warn(
                f"{filename} not found in {self.model_dir}. Using initialized weights."
            )
        model.eval()
        return model

    def _load_voices(self) -> Dict[str, np.ndarray]:
        if self.voices_path.exists():
            data = json.loads(self.voices_path.read_text())
            return {k: np.array(v, dtype=np.float32) for k, v in data.items()}
        return {}

    def save_voices(self):
        data = {k: v.tolist() for k, v in self.voices.items()}
        self.voices_path.write_text(json.dumps(data, indent=2))

    @torch.inference_mode()
    def clone_voice(self, audio_path: str) -> np.ndarray:
        wav, sr = torchaudio.load(audio_path)
        if sr != 16_000:
            wav = torchaudio.transforms.Resample(sr, 16_000)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        target_len = 16_000 * 3
        if wav.shape[1] > target_len:
            wav = wav[:, :target_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
        emb = self.speaker_encoder(wav.to(self.device))
        return emb.cpu().numpy()[0]

    def _chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) < self.max_chars:
                current += " " + sent
            else:
                if current:
                    chunks.append(current.strip())
                current = sent
        if current:
            chunks.append(current.strip())
        if not chunks:
            chunks = [text]
        return chunks

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speaker_emb: Optional[np.ndarray] = None,
        voice_id: Optional[str] = None,
        use_faraday: bool = True,
        use_aether: bool = True,
    ) -> np.ndarray:
        if speaker_emb is None and voice_id is not None:
            speaker_emb = self.voices.get(voice_id)
            if speaker_emb is None:
                raise ValueError(
                    f"Voice '{voice_id}' not found. Available: {list(self.voices.keys())}"
                )

        if speaker_emb is None:
            speaker_emb = np.zeros(192, dtype=np.float32)

        speaker_emb_t = (
            torch.from_numpy(speaker_emb).float().unsqueeze(0).to(self.device)
        )

        chunks = self._chunk_text(text)
        waveforms: List[np.ndarray] = []

        # Build SpeechT5 speaker embedding from cloned voice
        # speaker_emb_t is always provided (defaults to zeros), project to 512-dim
        if speaker_emb_t.shape[1] == 192:
            spk_for_tts = self.spk_proj_faraday(speaker_emb_t)
            spk_aether_base = self.spk_proj_aether(spk_for_tts)
        else:
            spk_for_tts = self.default_speaker
            spk_aether_base = torch.zeros(1, 192, device=self.device)

        for chunk in chunks:
            inputs = self.processor(text=chunk, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # SpeechT5 → spectrogram [T, 80] using ACTUAL speaker embedding
            spectrogram = self.tts.generate_speech(inputs["input_ids"], spk_for_tts)
            # Reshape to [1, 80, T] for Faraday
            mel = spectrogram.T.unsqueeze(0)  # [1, 80, T]

            # Faraday enhancement (supervised mode)
            spk_faraday = spk_for_tts  # already 512-dim
            if use_faraday:
                mel = mel.unsqueeze(1)  # [1, 1, 80, T]
                text_emb = torch.zeros(1, 512, device=self.device)
                # Use PERSISTENT projection layer only for 192-dim embeddings
                if speaker_emb_t.shape[1] == 192:
                    spk_faraday = self.spk_proj_faraday(speaker_emb_t)
                mel_enhanced = self.faraday.supervised_enhance(
                    mel, text_emb=text_emb, speaker_emb=spk_faraday
                )
                mel = mel_enhanced.squeeze(1)  # [1, 80, T]

            # Vocoder → waveform @ 16kHz
            wav_16k = self.vocoder(mel.squeeze(0).T)  # [T_samples]

            # Resample to 24kHz for Aether using PERSISTENT resample
            wav_24k = self.resample_16to24(wav_16k)

            # Aether waveform polish
            if use_aether:
                wav_t = wav_24k.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, T]
                aether_mel = self.mel_transform(wav_t.squeeze(1))  # [1, 80, T_mel]
                aether_mel = torch.log(aether_mel + 1e-6)
                T_mel = aether_mel.shape[2]
                energy = aether_mel.mean(dim=1, keepdim=True)  # [1, 1, T]
                f0 = torch.zeros(1, 1, T_mel, device=self.device)
                # Use PERSISTENT projection layer
                spk_aether = self.spk_proj_aether(spk_faraday)
                wav_refined = self.aether(wav_t, aether_mel, spk_aether, f0, energy)
                wav_24k = wav_refined[0, 0]

            waveforms.append(wav_24k.cpu().numpy())

        if not waveforms:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(waveforms)

    def save_wav(self, wav: np.ndarray, path: str, sample_rate: int = 24_000):
        sf.write(path, wav, sample_rate)
        print(f"[DemonTTS] Saved: {path}")

    def save_audio(self, wav: np.ndarray, path: str, sample_rate: int = 24_000):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        file_format = "FLAC" if output.suffix.lower() == ".flac" else None
        sf.write(output, wav, sample_rate, format=file_format)
        print(f"[DemonTTS] Saved: {output}")

    def combine_chapters(
        self,
        chapter_paths: List[str],
        output_path: str,
        pause_seconds: float = 1.0,
        sample_rate: int = 24_000,
    ):
        parts = []
        pause = np.zeros(int(pause_seconds * sample_rate), dtype=np.float32)
        for chapter_path in chapter_paths:
            wav, sr = sf.read(chapter_path, dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != sample_rate:
                wav_t = torch.from_numpy(np.atleast_1d(wav)).float().unsqueeze(0)
                wav = (
                    torchaudio.transforms.Resample(sr, sample_rate)(wav_t)
                    .squeeze(0)
                    .numpy()
                )
            parts.extend([wav.astype(np.float32), pause])
        combined = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        self.save_audio(combined, output_path, sample_rate=sample_rate)


def main():
    tts = DemonTTS()
    print(f"Faraday params:        {faraday_count(tts.faraday):,}")
    print(f"Aether params:         {aether_count(tts.aether.filter_net):,}")
    print(f"SpeakerEncoder params: {spk_count(tts.speaker_encoder):,}")

    wav = tts.synthesize(
        "Hello world. This is a test of the demon speech engine.",
        voice_id="Rick C-137",
    )
    print(f"Synthesized {len(wav)} samples ({len(wav) / 24_000:.2f}s)")
    tts.save_wav(wav, "test_output.wav")


if __name__ == "__main__":
    main()
