"""DemonTTS inference engine – wraps student, faraday, aether, vocoder."""

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
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from aether.model import AetherFilter, count_parameters as aether_count
from faraday.model import FaradayDiffusion, count_parameters as faraday_count
from neural.speaker_encoder import SpeakerEncoder, count_parameters as spk_count
from neural.student import StudentTTS, count_parameters as student_count
from neural.vocoder import HiFiGenerator, count_parameters as vocoder_count


class SimpleTokenizer:
    """BPE tokenizer with 10k vocab. Creates a basic one on first run."""

    def __init__(self, vocab_size: int = 10_000, model_path: str = "./models/tokenizer.json"):
        self.vocab_size = vocab_size
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.pad_id = 0
        self.unk_id = 1
        self.eos_id = 2
        self._load_or_create()

    def _load_or_create(self):
        if self.model_path.exists():
            self.tokenizer = Tokenizer.from_file(str(self.model_path))
            return

        print("[Tokenizer] No tokenizer found – training a minimal BPE model...")
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder()

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<pad>", "<unk>", "<eos>", "<s>"],
        )
        corpus = [
            "The quick brown fox jumps over the lazy dog. " * 200,
            "In the beginning God created the heaven and the earth. " * 100,
            "To be or not to be, that is the question. " * 100,
            "It was the best of times, it was the worst of times. " * 100,
            "Call me Ishmael. Some years ago never mind how long precisely having little or no money in my purse. " * 50,
            "Hello world. This is a test of the emergency broadcast system. " * 100,
            "Artificial intelligence is transforming the way we interact with technology. " * 80,
            "The speaker said Hello everyone. Welcome to the conference. " * 50,
        ]
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        tokenizer.save(str(self.model_path))
        self.tokenizer = tokenizer
        print(f"[Tokenizer] Saved to {self.model_path} (vocab={len(tokenizer.get_vocab())})")

    def encode(self, text: str) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        return [min(i, self.vocab_size - 1) for i in ids]


class GriffinLimVocoder(torch.nn.Module):
    """Griffin-Lim fallback vocoder (mel -> waveform)."""

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            n_iter=32,
        )

    @torch.inference_mode()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        linear = self.inverse_mel(mel)  # [B, n_fft//2+1, T]
        return self.griffin_lim(linear)  # [B, samples]


class MelVocoder(torch.nn.Module):
    """HiFi-GAN if checkpoint exists, else Griffin-Lim."""

    def __init__(self, model_dir: str = "./models", device: str = "cpu"):
        super().__init__()
        self.device = device
        self.use_hifigan = False
        hifigan_path = Path(model_dir) / "hifigan.pt"
        if hifigan_path.exists():
            try:
                self.hifigan = HiFiGenerator().to(device)
                state = torch.load(hifigan_path, map_location=device, weights_only=True)
                self.hifigan.load_state_dict(state)
                self.hifigan.eval()
                self.use_hifigan = True
                print("[MelVocoder] Loaded HiFi-GAN checkpoint")
            except Exception as exc:
                warnings.warn(f"HiFi-GAN load failed: {exc}. Using Griffin-Lim fallback.")
        if not self.use_hifigan:
            self.fallback = GriffinLimVocoder().to(device)
            print("[MelVocoder] Using Griffin-Lim fallback")

    @torch.inference_mode()
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if self.use_hifigan:
            return self.hifigan(mel)  # [B, samples]
        return self.fallback(mel)


class DemonTTS:
    """End-to-end TTS pipeline."""

    def __init__(
        self,
        model_dir: str = "./models",
        device: str = None,
        max_tokens: int = 512,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.max_tokens = max_tokens
        print(f"[DemonTTS] Device: {self.device}")

        self.student = self._load_model("student.pt", StudentTTS)
        self.speaker_encoder = self._load_model("speaker_encoder.pt", SpeakerEncoder)
        self.faraday = self._load_model("faraday.pt", FaradayDiffusion)
        self.aether = self._load_model("aether.pt", AetherFilter)
        self.vocoder = MelVocoder(model_dir=model_dir, device=self.device).to(self.device)

        # Project 192-dim speaker emb to 256-dim for Faraday conditioning
        self.spk_proj_faraday = torch.nn.Linear(192, 256).to(self.device)

        self.tokenizer = SimpleTokenizer(model_path=str(self.model_dir / "tokenizer.json"))

        self.voices_path = self.model_dir / "voices.json"
        self.voices: Dict[str, np.ndarray] = self._load_voices()

    def _load_model(self, filename: str, cls):
        path = self.model_dir / filename
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
            if len(current) + len(sent) < self.max_tokens * 4:
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

        spk_faraday = self.spk_proj_faraday(speaker_emb_t)  # [1, 256]

        chunks = self._chunk_text(text)
        waveforms: List[np.ndarray] = []

        for chunk in chunks:
            tokens = self.tokenizer.encode(chunk)
            if not tokens:
                continue
            tokens = tokens[: self.max_tokens]
            token_tensor = torch.tensor([tokens], dtype=torch.long, device=self.device)

            # Student → mel
            mel = self.student(token_tensor, speaker_emb_t)  # [1, 80, T]

            # Faraday diffusion enhancement
            mel = mel.unsqueeze(1)  # [1, 1, 80, T]
            text_emb = torch.zeros(1, 512, device=self.device)
            mel_enhanced = self.faraday.enhance(
                mel, text_emb=text_emb, speaker_emb=spk_faraday, steps=10
            )
            mel_enhanced = mel_enhanced.squeeze(1)  # [1, 80, T]

            # Vocoder
            wav = self.vocoder(mel_enhanced)  # [1, samples]
            waveforms.append(wav.cpu().numpy()[0])

        if not waveforms:
            return np.zeros(0, dtype=np.float32)

        full_wav = np.concatenate(waveforms)

        # Aether filter on full waveform
        wav_t = (
            torch.from_numpy(full_wav)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

        # Compute mel / f0 / energy for Aether
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=24_000,
            n_fft=1024,
            hop_length=256,
            n_mels=80,
        ).to(self.device)
        aether_mel = mel_tf(wav_t.squeeze(1))  # [B, 80, T_mel]
        aether_mel = torch.log(aether_mel + 1e-6)
        T_mel = aether_mel.shape[2]

        # Energy = mean mel energy per frame
        energy = aether_mel.mean(dim=1, keepdim=True)  # [1, 1, T]
        # F0 placeholder (zeros – Aether is untrained so this is fine for demo)
        f0 = torch.zeros(1, 1, T_mel, device=self.device)

        refined = self.aether(wav_t, aether_mel, speaker_emb_t, f0, energy)
        return refined.cpu().numpy()[0, 0]

    def save_wav(self, wav: np.ndarray, path: str, sample_rate: int = 24_000):
        sf.write(path, wav, sample_rate)
        print(f"[DemonTTS] Saved: {path}")


def main():
    tts = DemonTTS()
    print(f"Student params:       {student_count(tts.student):,}")
    print(f"SpeakerEncoder params: {spk_count(tts.speaker_encoder):,}")
    print(f"Faraday params:        {faraday_count(tts.faraday):,}")
    print(f"Aether params:         {aether_count(tts.aether.filter_net):,}")

    wav = tts.synthesize("Hello world. This is a test of the demon speech engine.")
    print(f"Synthesized {len(wav)} samples ({len(wav) / 24_000:.2f}s)")
    tts.save_wav(wav, "test_output.wav")


if __name__ == "__main__":
    main()
