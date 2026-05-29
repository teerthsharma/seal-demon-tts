"""Generate self-improved training data using trained DemonTTS.

Loads the current best Faraday+Aether models and uses them as the "teacher"
to generate higher-quality synthetic (input, target) pairs. The model teaches
itself by producing better targets than the original SpeechT5 teacher.
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import SpeechT5ForTextToSpeech, SpeechT5HifiGan, SpeechT5Processor

from aether.model import AetherFilter
from faraday.model import FaradayDiffusion

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_models(models_dir: str, device: torch.device):
    """Load trained Faraday and Aether models."""
    model_path = Path(models_dir)

    faraday = FaradayDiffusion(text_dim=512, speaker_dim=512, cond_dim=512, base_channels=192).to(device)
    aether = AetherFilter().to(device)

    faraday_ckpt = model_path / "faraday.pt"
    aether_ckpt = model_path / "aether.pt"

    if faraday_ckpt.exists():
        faraday.load_state_dict(torch.load(faraday_ckpt, map_location=device, weights_only=True))
        print(f"[OuroborosData] Loaded Faraday from {faraday_ckpt}")
    else:
        print(f"[OuroborosData] WARNING: {faraday_ckpt} not found, using random init")

    if aether_ckpt.exists():
        aether.load_state_dict(torch.load(aether_ckpt, map_location=device, weights_only=True))
        print(f"[OuroborosData] Loaded Aether from {aether_ckpt}")
    else:
        print(f"[OuroborosData] WARNING: {aether_ckpt} not found, using random init")

    faraday.eval()
    aether.eval()
    return faraday, aether


def synthesize_teacher(text, processor, tts, vocoder, speaker_emb, device):
    """Run SpeechT5 text → spectrogram → waveform."""
    inputs = processor(text=text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] > 600:
        input_ids = input_ids[:, :600]

    with torch.no_grad():
        spectrogram = tts.generate_speech(input_ids, speaker_emb)
        wav = vocoder(spectrogram)
    return spectrogram.cpu(), wav.cpu()


def corrupt_mel(mel):
    """Add synthetic corruption for Faraday training pairs."""
    corrupted = mel.clone()
    noise = torch.randn_like(corrupted) * corrupted.std() * 0.15
    corrupted = corrupted + noise
    num_bins = corrupted.shape[1]
    num_mask = random.randint(0, int(num_bins * 0.15))
    if num_mask > 0:
        mask_bins = random.sample(range(num_bins), num_mask)
        corrupted[:, mask_bins] = 0
    return corrupted


def corrupt_waveform(wav, sr=16000):
    """Add synthetic corruption for Aether training pairs."""
    corrupted = wav.clone()
    noise = torch.randn_like(corrupted) * corrupted.std() * 0.10
    corrupted = corrupted + noise
    if random.random() < 0.4:
        target_sr = random.choice([8000, 12000])
        resampled = torchaudio.transforms.Resample(sr, target_sr)(corrupted)
        corrupted = torchaudio.transforms.Resample(target_sr, sr)(resampled)
    return corrupted


def generate_pairs(
    text_chunks,
    processor,
    tts,
    vocoder,
    faraday,
    aether,
    output_dir,
    num_pairs,
    device,
):
    """Generate self-improved training pairs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "faraday_pairs").mkdir(exist_ok=True)
    (output_dir / "aether_pairs").mkdir(exist_ok=True)

    speaker_emb = torch.randn(1, 512).to(device)

    for pair_idx in tqdm(range(num_pairs), desc="Generating Ouroboros data"):
        text = random.choice(text_chunks)

        spectrogram, wav = synthesize_teacher(text, processor, tts, vocoder, speaker_emb, device)
        mel = spectrogram.T.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, 80, T]

        # Faraday: corrupt → enhance
        corrupted_mel = corrupt_mel(mel.squeeze(0).squeeze(0)).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            text_emb = torch.zeros(1, 512, device=device)
            spk_faraday = speaker_emb
            enhanced_mel = faraday.supervised_enhance(corrupted_mel, text_emb=text_emb, speaker_emb=spk_faraday)

        faraday_pair = {
            "student_mel": corrupted_mel.cpu(),
            "gt_mel": enhanced_mel.cpu(),
            "text_emb": text_emb.cpu().squeeze(0),
            "speaker_emb": spk_faraday.cpu().squeeze(0),
        }
        torch.save(faraday_pair, output_dir / "faraday_pairs" / f"pair_{pair_idx:06d}.pt")

        # Aether: corrupt waveform → enhance
        wav_24k = torchaudio.transforms.Resample(16000, 24000)(wav)
        corrupted_wav = corrupt_waveform(wav_24k, sr=24000)
        mel_aether = torchaudio.transforms.MelSpectrogram(
            sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80
        )(corrupted_wav.unsqueeze(0))
        mel_aether = torch.log(mel_aether + 1e-6).to(device)
        T_mel = mel_aether.shape[2]
        energy = mel_aether.mean(dim=1, keepdim=True)
        f0 = torch.zeros(1, 1, T_mel, device=device)
        spk_aether = torch.randn(1, 192, device=device)
        wav_in = corrupted_wav.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            enhanced_wav = aether(wav_in, mel_aether, spk_aether, f0, energy)

        aether_pair = {
            "input_waveform": corrupted_wav.unsqueeze(0).cpu(),
            "target_waveform": enhanced_wav.cpu(),
            "mel": mel_aether.cpu(),
            "speaker_emb": spk_aether.cpu().squeeze(0),
            "f0": f0.cpu().squeeze(0),
            "energy": energy.cpu().squeeze(0),
        }
        torch.save(aether_pair, output_dir / "aether_pairs" / f"pair_{pair_idx:06d}.pt")

    print(f"[OuroborosData] Generated {num_pairs} pairs in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./book_parsed/")
    parser.add_argument("--output_dir", default="./ouroboros_generated/")
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--models_dir", default="./models/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[OuroborosData] Device: {device}")

    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    tts = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts", use_safetensors=True).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan", use_safetensors=True).to(device)
    tts.eval()
    vocoder.eval()

    faraday, aether = load_models(args.models_dir, device)

    text_chunks = []
    for json_path in Path(args.input_dir).glob("*.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            book = json.load(f)
        for chapter in book.values():
            text = chapter.get("text", "")
            sentences = text.replace("\n", " ").split(". ")
            for sent in sentences:
                sent = sent.strip()
                if 20 < len(sent) < 400:
                    text_chunks.append(sent)

    if not text_chunks:
        raise ValueError(f"No text chunks found in {args.input_dir}")

    print(f"[OuroborosData] Loaded {len(text_chunks)} text chunks")

    generate_pairs(
        text_chunks,
        processor,
        tts,
        vocoder,
        faraday,
        aether,
        args.output_dir,
        args.num_pairs,
        device,
    )


if __name__ == "__main__":
    main()
