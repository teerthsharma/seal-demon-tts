"""Transformer-based filter coefficient predictor + end-to-end filter net.

Scales from 0.8M LSTM to 100M+ parameter transformer for massive
expression capacity in waveform refinement.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from aether.lattice_filter import LatticeFilterBank, count_parameters


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + attn_out
        # FFN with pre-norm
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class FilterNet(nn.Module):
    """100M-parameter transformer-based filter coefficient predictor.

    Predicts reflection coefficients from mel + speaker + prosody using
    deep self-attention over time, then applies a parallel lattice filter bank.
    """

    def __init__(
        self,
        mel_bins: int = 80,
        speaker_dim: int = 192,
        hidden: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        n_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden = hidden
        self.n_channels = n_channels

        # Deep feature projections
        self.mel_proj = nn.Sequential(
            nn.Conv1d(mel_bins, hidden, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )
        self.f0_proj = nn.Sequential(
            nn.Conv1d(1, hidden // 4, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden // 4, hidden // 4, kernel_size=3, padding=1),
        )
        self.energy_proj = nn.Sequential(
            nn.Conv1d(1, hidden // 4, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.Conv1d(hidden // 4, hidden // 4, kernel_size=3, padding=1),
        )

        proj_total = hidden + hidden // 4 + hidden // 4
        self.speaker_proj = nn.Sequential(
            nn.Linear(speaker_dim, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, proj_total),
        )

        # Input projection to transformer dim
        self.input_proj = nn.Linear(proj_total, hidden)

        # Positional encoding
        self.pos_emb = SinusoidalPosEmb(hidden)

        # Deep transformer stack
        self.transformer = nn.ModuleList([
            TransformerBlock(hidden, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output: reflection coefficients per channel
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_channels),
        )

        # Differentiable lattice filter bank
        self.filter_bank = LatticeFilterBank(n_channels)

    def forward(
        self,
        waveform: torch.Tensor,
        mel: torch.Tensor,
        speaker_emb: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            waveform: [B, 1, T_wav]
            mel: [B, 80, T_mel]
            speaker_emb: [B, 192]
            f0: [B, 1, T_mel]
            energy: [B, 1, T_mel]
        Returns:
            [B, 1, T_wav] refined waveform
        """
        B = mel.size(0)
        T_mel = mel.size(2)

        # Project features
        m = self.mel_proj(mel)          # [B, hidden, T_mel]
        f = self.f0_proj(f0)            # [B, hidden/4, T_mel]
        e = self.energy_proj(energy)    # [B, hidden/4, T_mel]
        x = torch.cat([m, f, e], dim=1)  # [B, hidden*1.5, T_mel]
        x = x.transpose(1, 2)            # [B, T_mel, hidden*1.5]

        # Add speaker conditioning
        spk = self.speaker_proj(speaker_emb)[:, None, :]  # [B, 1, hidden*1.5]
        x = x + spk

        # Project to transformer dimension
        x = self.input_proj(x)  # [B, T_mel, hidden]

        # Add positional encoding
        pos = self.pos_emb(torch.arange(T_mel, device=x.device))  # [T_mel, hidden]
        x = x + pos[None, :, :]

        # Apply transformer layers
        for block in self.transformer:
            x = block(x)

        # Output reflection coefficients
        x = self.out_norm(x)
        coeffs = self.out(x)  # [B, T_mel, n_channels]
        coeffs = coeffs.transpose(1, 2)  # [B, n_channels, T_mel]

        # Upsample coefficients to waveform length
        T_wav = waveform.size(2)
        coeffs = F.interpolate(coeffs, size=T_wav, mode="linear", align_corners=False)

        # Apply through lattice filter bank
        out = self.filter_bank(waveform, coeffs)
        return out


if __name__ == "__main__":
    net = FilterNet()
    total = count_parameters(net)
    print(f"[FilterNet] Params: {total:,} (~{total/1e6:.1f}M)")
    print(f"Target: ~100M")

    wav = torch.randn(1, 1, 24000)
    mel = torch.randn(1, 80, 100)
    spk = torch.randn(1, 192)
    f0 = torch.randn(1, 1, 100)
    energy = torch.randn(1, 1, 100)
    out = net(wav, mel, spk, f0, energy)
    print(f"Output shape: {out.shape}")

    # Memory footprint
    print(f"Estimated fp16 memory: {total * 2 / 1e6:.2f} MB")
