"""Multi-resolution STFT loss + perceptual loss for Aether filter training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T


class MRSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (L1 magnitude + L1 log-magnitude)."""

    def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(128, 256, 512)):
        super().__init__()
        self.stfts = nn.ModuleList([
            T.Spectrogram(n_fft=n, hop_length=h, power=1.0, normalized=True)
            for n, h in zip(fft_sizes, hop_sizes)
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred/target: [B, 1, T] or [B, T]
        if pred.dim() == 3:
            pred = pred.squeeze(1)
            target = target.squeeze(1)
        loss = 0.0
        for stft in self.stfts:
            sp = stft(pred)
            st = stft(target)
            loss += F.l1_loss(sp, st)
            loss += F.l1_loss(torch.log(sp + 1e-5), torch.log(st + 1e-5))
        return loss


class PerceptualLoss(nn.Module):
    """Mel-spectrogram L1 as a lightweight perceptual proxy."""

    def __init__(self, sample_rate: int = 24000, n_mels: int = 80):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.squeeze(1)
            target = target.squeeze(1)
        mp = torch.log(self.mel(pred) + 1e-5)
        mt = torch.log(self.mel(target) + 1e-5)
        return F.l1_loss(mp, mt)


class TotalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mr_stft = MRSTFTLoss()
        self.perceptual = PerceptualLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mr_stft(pred, target) + 0.5 * self.perceptual(pred, target)
