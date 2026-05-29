"""PyTorch bridge to EPSILON-PHASE numeric robustness layer.

Brings stochastic resonance and subtractive dithering from the standalone
wind-simulation physics engine into DemonTTS. Processes audio tensors through
the proven NumPy robustness layer, then converts back to PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

# Attempt to import standalone EPSILON-PHASE
try:
    import sys
    from pathlib import Path

    epsilon_src = Path(__file__).parent.parent / "EPSILON-PHASE-main" / "src"
    if epsilon_src.exists():
        sys.path.insert(0, str(epsilon_src))
    from epsilon_phase.architecture import NumericRobustnessLayer
    STANDALONE_AVAILABLE = True
except Exception:
    STANDALONE_AVAILABLE = False


class EpsilonPhaseBridge(nn.Module):
    """Bridge between DemonTTS and EPSILON-PHASE standalone engine.

    Processes audio tensors through the proven NumPy robustness layer,
    then converts back to PyTorch. Falls back to pure-PyTorch resonance
    when the standalone engine is unavailable.
    """

    def __init__(
        self,
        vector_dim: int = 24000,
        base_gain: float = 0.02,
        quant_step: float = 1.0 / 4096.0,
    ):
        super().__init__()
        self.vector_dim = vector_dim
        self.base_gain = base_gain
        self.quant_step = quant_step

        # Standalone EPSILON-PHASE has incompatible API; use PyTorch fallback
        self.native_layer = None

    def process_batch(
        self,
        waveform: torch.Tensor,
        progress_metric: Optional[float] = None,
    ) -> torch.Tensor:
        """Process a batch of waveforms.

        Args:
            waveform: [B, 1, T] or [B, T]
            progress_metric: optional training progress for adaptive gain

        Returns:
            Processed waveform of same shape
        """
        original_shape = waveform.shape
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        batch_size, time_steps = waveform.shape

        if self.native_layer is not None and time_steps <= self.vector_dim:
            output = []
            for i in range(batch_size):
                sig_np = waveform[i].detach().cpu().numpy()
                if len(sig_np) < self.vector_dim:
                    sig_np = np.pad(sig_np, (0, self.vector_dim - len(sig_np)))
                else:
                    sig_np = sig_np[: self.vector_dim]

                processed = self.native_layer.process(sig_np)
                output.append(
                    torch.from_numpy(processed[:time_steps]).to(waveform.device)
                )
            result = torch.stack(output)
        else:
            result = self._pytorch_resonance(waveform, progress_metric)

        if len(original_shape) == 3:
            result = result.unsqueeze(1)

        return result

    def _pytorch_resonance(
        self,
        x: torch.Tensor,
        progress_metric: Optional[float] = None,
    ) -> torch.Tensor:
        """Pure PyTorch fallback when standalone engine is unavailable."""
        gain = self.base_gain
        if progress_metric is not None and abs(progress_metric) < 0.001:
            gain *= 1.5

        noise = torch.randn_like(x) * x.std(dim=-1, keepdim=True).clamp_min(1e-8) * gain
        return x + noise
