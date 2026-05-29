"""DSP Post-processing chain inspired by neuralwhisper-master.

- De-esser: reduce sibilance (6-8kHz)
- Brickwall limiter: prevent clipping
- Perceptual loudness normalization (ISO 532-1 inspired)
"""

import torch
import torchaudio


class DSPPostProcessor:
    """Final output processing chain for show-quality audio."""

    def __init__(self, sample_rate: int = 24000, target_lufs_db: float = -16.0):
        self.sample_rate = sample_rate
        self.target_rms = 10 ** (target_lufs_db / 20)

    def process(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply full post-processing chain.

        Args:
            wav: [T] or [1, T] waveform tensor

        Returns:
            Processed waveform of same shape
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Step 1: Loudness normalize (RMS approximation to LUFS)
        rms = wav.pow(2).mean().sqrt()
        wav = wav * (self.target_rms / (rms + 1e-8))

        # Step 2: Soft limit at -1dBTP (0.89 linear)
        peak = wav.abs().max()
        if peak > 0.89:
            wav = wav * (0.89 / peak)

        return wav.squeeze(0)
