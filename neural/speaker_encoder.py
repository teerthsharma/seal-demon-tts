"""ECAPA-TDNN speaker encoder wrapper for zero-shot cloning."""

import torch
import torch.nn as nn
import torchaudio


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SEBlock(nn.Module):
    """Squeeze-Excitation block."""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        b, c, _ = x.shape
        w = x.mean(dim=2)  # [B, C]
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w[:, :, None]


class TDNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=dilation * (kernel_size // 2), dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.se(x)
        return x


class ECAPAModel(nn.Module):
    """Simplified ECAPA-TDNN (~10M params)."""

    def __init__(self, input_dim: int = 80, emb_dim: int = 192, channels: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)

        self.tdnn2 = TDNNBlock(channels, channels, 3, 2)
        self.tdnn3 = TDNNBlock(channels, channels, 3, 3)
        self.tdnn4 = TDNNBlock(channels, channels, 3, 4)
        self.tdnn5 = TDNNBlock(channels, channels, 3, 5)

        self.attention = nn.Sequential(
            nn.Conv1d(channels * 4, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, channels * 4, 1),
            nn.Softmax(dim=2),
        )
        self.bn_agg = nn.BatchNorm1d(channels * 4 * 2)
        self.fc = nn.Linear(channels * 4 * 2, emb_dim)
        self.bn_emb = nn.BatchNorm1d(emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, input_dim, T] (mel-spectrogram)
        x = torch.relu(self.bn1(self.conv1(x)))
        x2 = self.tdnn2(x)
        x3 = self.tdnn3(x2)
        x4 = self.tdnn4(x3)
        x5 = self.tdnn5(x4)
        x = torch.cat([x2, x3, x4, x5], dim=1)

        w = self.attention(x)
        mu = (x * w).sum(dim=2)
        sg = torch.sqrt(((x - mu[:, :, None]) ** 2 * w).sum(dim=2).clamp(min=1e-5))
        x = torch.cat([mu, sg], dim=1)
        x = self.bn_agg(x)
        x = self.fc(x)
        x = self.bn_emb(x)
        return x


class SpeakerEncoder(nn.Module):
    """Wrapper that loads audio, computes mel, and returns speaker embedding."""

    def __init__(self, emb_dim: int = 192, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.emb_dim = emb_dim
        self.model = ECAPAModel(input_dim=80, emb_dim=emb_dim)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=400,
            hop_length=160,
            n_mels=80,
        )

    def preprocess(self, audio_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
        # mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        # trim/pad to exactly 3 seconds
        target_len = self.sample_rate * 3
        if wav.shape[1] > target_len:
            wav = wav[:, :target_len]
        else:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))
        return wav

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: [B, 1, T] or [B, T]
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)  # [B, T]
        mel = self.mel_transform(waveform)  # [B, 80, T_mel]
        mel = torch.log(mel + 1e-6)
        emb = self.model(mel)
        return emb


if __name__ == "__main__":
    enc = SpeakerEncoder()
    print(f"[SpeakerEncoder] Params: {count_parameters(enc):,}")
    wav = torch.randn(2, 1, 48_000)
    emb = enc(wav)
    print(f"Embedding shape: {emb.shape}")
