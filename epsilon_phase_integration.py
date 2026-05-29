#!/usr/bin/env python3
"""
EPSILON-PHASE Integration for DemonTTS

Brings numeric robustness, stochastic resonance, and Lyapunov-governed
noise injection into the Faraday diffusion and Aether filtering pipeline.

The core insight from EPSILON-PHASE: quantization noise and numeric
instability in diffusion can be tamed by controlled stochastic resonance.
When progress stagnates, inject structured noise. When progress flows,
let the signal dominate.

Author: Seal — because diffusion without numeric robustness is just chaos.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class EpsilonPhaseConfig:
    """Configuration for EPSILON-PHASE numeric robustness layer."""
    # Stochastic resonance
    base_gain: float = 0.02
    min_gain: float = 0.005
    max_gain: float = 0.30
    stagnation_threshold: float = 1e-3
    gain_growth: float = 1.15
    gain_decay: float = 0.96

    # Subtractive dithering
    quant_step: float = 1.0 / 4096.0
    dither_seed: int = 42

    # Lyapunov governor (for diffusion sigma)
    base_sigma: float = 0.02
    min_sigma: float = 0.005
    max_sigma: float = 0.30
    flatten_threshold: float = 1e-3
    sigma_growth: float = 1.15
    sigma_decay: float = 0.96

    # SDE dynamics
    dt: float = 1.0
    state_norm_limit: float = 8.0


class LyapunovGovernor(nn.Module):
    """PyTorch Lyapunov governor for diffusion sigma adaptation.

    Adapts noise sigma based on local loss slope. When loss stops
    improving, sigma grows (more exploration). When loss drops,
    sigma decays (more exploitation).
    """

    def __init__(self, config: Optional[EpsilonPhaseConfig] = None):
        super().__init__()
        self.config = config or EpsilonPhaseConfig()
        self.register_buffer('_sigma', torch.tensor(self.config.base_sigma))
        self.register_buffer('_last_loss', torch.tensor(float('nan')))

    @property
    def sigma(self) -> float:
        return float(self._sigma.item())

    def update(self, loss: Optional[float] = None) -> float:
        if loss is None:
            return self.sigma

        loss_t = float(loss)
        if math.isnan(self._last_loss.item()):
            self._last_loss.fill_(loss_t)
            return self.sigma

        slope = abs(loss_t - self._last_loss.item())
        self._last_loss.fill_(loss_t)

        if slope < self.config.flatten_threshold:
            # Loss stagnated — increase sigma (more noise = more exploration)
            new_sigma = min(self.config.max_sigma, self.sigma * self.config.sigma_growth)
        else:
            # Loss improving — decrease sigma (less noise = more exploitation)
            new_sigma = max(self.config.min_sigma, self.sigma * self.config.sigma_decay)

        self._sigma.fill_(new_sigma)
        return new_sigma


class StochasticResonanceLayer(nn.Module):
    """PyTorch stochastic resonance injection for tensor signal paths.

    When training progress stagnates, injects normalized noise into
    the signal with adaptive gain. This helps escape local minima
    in the diffusion landscape.
    """

    def __init__(self, config: Optional[EpsilonPhaseConfig] = None):
        super().__init__()
        self.config = config or EpsilonPhaseConfig()
        self.register_buffer('_gain', torch.tensor(self.config.base_gain))
        self.register_buffer('_last_metric', torch.tensor(float('nan')))

    @property
    def gain(self) -> float:
        return float(self._gain.item())

    def update_gain(self, progress_metric: Optional[float] = None) -> float:
        if progress_metric is None:
            return self.gain

        m_t = float(progress_metric)
        if math.isnan(self._last_metric.item()):
            self._last_metric.fill_(m_t)
            return self.gain

        slope = abs(m_t - self._last_metric.item())
        self._last_metric.fill_(m_t)

        if slope < self.config.stagnation_threshold:
            new_gain = min(self.config.max_gain, self.gain * self.config.gain_growth)
        else:
            new_gain = max(self.config.min_gain, self.gain * self.config.gain_decay)

        self._gain.fill_(new_gain)
        return new_gain

    def forward(self, signal: torch.Tensor, noise: Optional[torch.Tensor] = None,
                progress_metric: Optional[float] = None) -> torch.Tensor:
        """Inject stochastic resonance into signal.

        Args:
            signal: [*, T] tensor signal
            noise: [*, T] optional noise tensor (generated if None)
            progress_metric: scalar progress metric for gain adaptation

        Returns:
            signal + gain * normalized_noise
        """
        if noise is None:
            noise = torch.randn_like(signal)

        # Normalize noise: zero mean, unit variance
        noise = noise - noise.mean(dim=-1, keepdim=True)
        std = noise.std(dim=-1, keepdim=True).clamp(min=1e-8)
        noise = noise / std

        gain = self.update_gain(progress_metric)
        return signal + gain * noise


class SubtractiveDitherLayer(nn.Module):
    """PyTorch subtractive dither for quantization decorrelation.

    Adds uniform dither before quantization, then subtracts it after.
    Reduces signal-correlated quantization error in mel spectrograms
    and filter coefficients.
    """

    def __init__(self, quant_step: float = 1.0 / 4096.0, seed: int = 42):
        super().__init__()
        self.quant_step = quant_step
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply subtractive dither quantization.

        Q(u) = Δ · round(u/Δ)
        y = Q(x + d) - d
        where d ~ U(-Δ/2, Δ/2)
        """
        dither = torch.rand(x.shape, generator=self.generator, device=x.device)
        dither = dither * self.quant_step - 0.5 * self.quant_step

        quant = torch.round((x + dither) / self.quant_step) * self.quant_step
        return quant - dither


class EpsilonPhaseDiffusion(nn.Module):
    """Faraday diffusion enhanced with EPSILON-PHASE numeric robustness.

    Replaces the standard noise schedule with a Lyapunov-governed
    adaptive sigma that responds to training loss dynamics.
    """

    def __init__(self, base_scheduler, config: Optional[EpsilonPhaseConfig] = None):
        super().__init__()
        self.base_scheduler = base_scheduler
        self.governor = LyapunovGovernor(config)
        self.resonance = StochasticResonanceLayer(config)
        self.dither = SubtractiveDitherLayer(
            quant_step=(config or EpsilonPhaseConfig()).quant_step
        )

    def add_noise(self, x: torch.Tensor, t: torch.Tensor,
                  loss_proxy: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise with Lyapunov-governed sigma.

        Args:
            x: clean signal [B, C, H, W]
            t: timestep indices [B]
            loss_proxy: optional loss value for sigma adaptation

        Returns:
            (noised_x, noise) with adaptive noise magnitude
        """
        # Adapt sigma based on loss dynamics
        sigma = self.governor.update(loss_proxy)

        # Standard noise
        noise = torch.randn_like(x)

        # Apply stochastic resonance: boost noise when stagnating
        noise = self.resonance(noise, progress_metric=loss_proxy)

        # Scale by adaptive sigma
        noised = x + sigma * noise

        # Subtractive dither on quantized mels
        noised = self.dither(noised)

        return noised, noise


class EpsilonPhaseAether(nn.Module):
    """Aether filter coefficients enhanced with EPSILON-PHASE dithering.

    Applies subtractive dither to predicted reflection coefficients
    before they drive the IIR lattice, reducing coefficient quantization
    artifacts that cause metallic ringing.
    """

    def __init__(self, config: Optional[EpsilonPhaseConfig] = None):
        super().__init__()
        self.config = config or EpsilonPhaseConfig()
        self.dither = SubtractiveDitherLayer(quant_step=self.config.quant_step)
        self.resonance = StochasticResonanceLayer(self.config)

    def refine_coefficients(self, coeffs: torch.Tensor,
                            progress_metric: Optional[float] = None) -> torch.Tensor:
        """Apply numeric robustness to filter coefficients.

        Args:
            coeffs: [B, n_channels, T] predicted reflection coefficients
            progress_metric: optional training loss for resonance adaptation

        Returns:
            dithered + resonance-enhanced coefficients
        """
        # Dither coefficients to decorrelate quantization error
        coeffs = self.dither(coeffs)

        # Add stochastic resonance when training stagnates
        coeffs = self.resonance(coeffs, progress_metric=progress_metric)

        # Ensure stability: reflection coefficients must be in [-1, 1]
        coeffs = torch.tanh(coeffs)

        return coeffs


class EpsilonPhaseTrainingWrapper:
    """Wraps training loops with EPSILON-PHASE monitoring.

    Tracks loss slope, reports sigma/gain dynamics, and provides
    early warnings for numeric instability.
    """

    def __init__(self, config: Optional[EpsilonPhaseConfig] = None):
        self.config = config or EpsilonPhaseConfig()
        self.governor = LyapunovGovernor(config)
        self.resonance = StochasticResonanceLayer(config)
        self.history: list = []

    def step(self, loss: float, step: int) -> Dict[str, float]:
        """Process one training step and return diagnostics.

        Args:
            loss: current step loss
            step: step number

        Returns:
            dict with sigma, gain, snr_estimate, stability_norm
        """
        sigma = self.governor.update(loss)
        gain = self.resonance.update_gain(loss)

        self.history.append({
            "step": step,
            "loss": loss,
            "sigma": sigma,
            "gain": gain,
        })

        return {
            "sigma": sigma,
            "gain": gain,
            "snr_db": 10 * math.log10(1.0 / (loss + 1e-12)),
            "stability_norm": math.exp(-loss),  # heuristic: lower loss = more stable
        }

    def report(self) -> str:
        """Generate a human-readable training dynamics report."""
        if not self.history:
            return "[EpsilonPhase] No training history yet."

        recent = self.history[-10:]
        avg_loss = sum(h["loss"] for h in recent) / len(recent)
        avg_sigma = sum(h["sigma"] for h in recent) / len(recent)
        avg_gain = sum(h["gain"] for h in recent) / len(recent)

        status = "EXPLORING" if avg_sigma > 0.15 else "EXPLOITING"

        lines = [
            "=" * 50,
            "  EPSILON-PHASE Training Dynamics Report",
            "=" * 50,
            f"  Status:        {status}",
            f"  Avg Loss:      {avg_loss:.6f}",
            f"  Avg Sigma:     {avg_sigma:.4f} (noise magnitude)",
            f"  Avg Gain:      {avg_gain:.4f} (resonance strength)",
            f"  Steps tracked: {len(self.history)}",
            "=" * 50,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("EPSILON-PHASE Integration for DemonTTS")
    print("=" * 60)

    # Test Lyapunov governor
    gov = LyapunovGovernor()
    print(f"\n[LyapunovGovernor] Initial sigma: {gov.sigma:.4f}")

    for loss in [1.0, 0.95, 0.90, 0.89, 0.89, 0.89, 0.85]:
        sigma = gov.update(loss)
        print(f"  Loss={loss:.2f} -> sigma={sigma:.4f}")

    # Test stochastic resonance
    sr = StochasticResonanceLayer()
    signal = torch.sin(torch.linspace(0, 4 * math.pi, 256))
    noise = torch.randn(256)
    result = sr(signal, noise, progress_metric=0.5)
    print(f"\n[StochasticResonance] Signal std: {signal.std():.4f}")
    print(f"[StochasticResonance] Output std:  {result.std():.4f}")

    # Test subtractive dither
    dither = SubtractiveDitherLayer(quant_step=1.0 / 4096.0)
    x = torch.randn(1000)
    y = dither(x)
    error = (x - y).abs().mean()
    print(f"\n[SubtractiveDither] Mean quantization error: {error:.6f}")

    # Test training wrapper
    wrapper = EpsilonPhaseTrainingWrapper()
    for step, loss in enumerate([1.0, 0.9, 0.8, 0.75, 0.74, 0.74, 0.73]):
        diag = wrapper.step(loss, step)
        print(f"[TrainingWrapper] Step {step}: loss={loss:.3f}, sigma={diag['sigma']:.4f}, gain={diag['gain']:.4f}")

    print("\n" + wrapper.report())
    print("\nEPSILON-PHASE integration ready. Rick can't break this one.")
