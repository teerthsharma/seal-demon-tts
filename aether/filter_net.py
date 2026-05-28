"""LSTM-based filter coefficient predictor + end-to-end filter net."""

import torch
import torch.nn as nn

from aether.lattice_filter import LatticeFilterBank, count_parameters


class FilterNet(nn.Module):
    """Predicts reflection coefficients from mel + speaker + prosody."""

    def __init__(
        self,
        mel_bins: int = 80,
        speaker_dim: int = 192,
        hidden: int = 128,
        n_layers: int = 2,
        n_channels: int = 64,
    ):
        super().__init__()
        self.mel_proj = nn.Conv1d(mel_bins, hidden, kernel_size=3, padding=1)
        self.f0_proj = nn.Conv1d(1, hidden // 4, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, hidden // 4, kernel_size=3, padding=1)
        lstm_input_size = hidden + hidden // 4 + hidden // 4
        self.speaker_proj = nn.Linear(speaker_dim, lstm_input_size)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.out = nn.Linear(hidden * 2, n_channels)
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

        m = self.mel_proj(mel)
        f = self.f0_proj(f0)
        e = self.energy_proj(energy)
        x = torch.cat([m, f, e], dim=1)  # [B, hidden*1.5, T_mel]
        x = x.transpose(1, 2)  # [B, T_mel, hidden*1.5]

        spk = self.speaker_proj(speaker_emb)[:, None, :]  # [B, 1, hidden]
        x = x + spk

        lstm_out, _ = self.lstm(x)  # [B, T_mel, hidden*2]
        coeffs = self.out(lstm_out)  # [B, T_mel, n_channels]
        coeffs = coeffs.transpose(1, 2)  # [B, n_channels, T_mel]

        # Upsample coefficients to waveform length
        T_wav = waveform.size(2)
        coeffs = torch.nn.functional.interpolate(coeffs, size=T_wav, mode="linear", align_corners=False)

        out = self.filter_bank(waveform, coeffs)
        return out


if __name__ == "__main__":
    net = FilterNet()
    print(f"[FilterNet] Params: {count_parameters(net):,}")
    wav = torch.randn(1, 1, 24000)
    mel = torch.randn(1, 80, 100)
    spk = torch.randn(1, 192)
    f0 = torch.randn(1, 1, 100)
    energy = torch.randn(1, 1, 100)
    out = net(wav, mel, spk, f0, energy)
    print(f"Output shape: {out.shape}")
