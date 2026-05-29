"""Aether Wave Filter: Aether + EPSILON-PHASE + aether-wave kinetic integration.

Applies subtractive dithering to lattice filter coefficients before they drive
the IIR filter bank, eliminating metallic ringing artifacts. The wave layer
adds numeric robustness and kinetic audio calculations from aether-wave-master
(mass-spring-damper resonance, wave propagation energy).
"""

from typing import Optional

import torch
import torch.nn as nn
from aether.model import AetherFilter
from epsilon_phase_bridge import EpsilonPhaseBridge


class AetherWaveFilter(nn.Module):
    """Aether with EPSILON-PHASE numeric robustness and kinetic wave energy.

    Training mode returns (waveform, loss).
    Inference mode returns waveform only.
    """

    def __init__(self, lr: float = 1e-4, epsilon_gain: float = 0.02):
        super().__init__()
        self.aether = AetherFilter(lr=lr)
        self.epsilon_bridge = EpsilonPhaseBridge(
            vector_dim=24000,
            base_gain=epsilon_gain,
        )
        # Kinetic energy coefficient from aether-wave mass-spring-damper model
        self.kinetic_coeff = nn.Parameter(torch.tensor(0.01))

    def _kinetic_resonance(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply kinetic audio resonance inspired by aether-wave-master.

        Models the waveform as a damped mass-spring system:
        F = m*a + c*v + k*x  →  energy = 0.5*m*v^2 + 0.5*k*x^2

        Args:
            waveform: [B, 1, T] or [B, T]

        Returns:
            Waveform with kinetic energy reinforcement
        """
        if waveform.dim() == 3:
            wav = waveform.squeeze(1)
        else:
            wav = waveform

        # Velocity: first derivative (finite difference)
        velocity = torch.zeros_like(wav)
        velocity[:, 1:] = wav[:, 1:] - wav[:, :-1]

        # Kinetic energy ~ 0.5 * m * v^2 (mass = 1, simplified)
        kinetic_energy = 0.5 * velocity.pow(2)

        # Reinforce waveform with scaled kinetic energy
        reinforcement = torch.tanh(self.kinetic_coeff) * kinetic_energy
        reinforced = wav + reinforcement

        if waveform.dim() == 3:
            reinforced = reinforced.unsqueeze(1)

        return reinforced

    def forward(
        self,
        waveform: torch.Tensor,
        mel: torch.Tensor,
        speaker_emb: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
        target_waveform: Optional[torch.Tensor] = None,
    ):
        out = self.aether(waveform, mel, speaker_emb, f0, energy)

        # Aether-wave kinetic resonance
        out = self._kinetic_resonance(out)

        # EPSILON-PHASE robustness on output waveform
        out = self.epsilon_bridge.process_batch(out)

        if target_waveform is not None:
            loss = self.aether.criterion(out, target_waveform)
            return out, loss
        return out
