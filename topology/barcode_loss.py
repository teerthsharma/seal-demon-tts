"""Topological loss for mel spectrograms using Betti number matching.

Penalizes differences in topological structure between predicted and target
spectrograms. This is the core of topology diffusion — the model must learn
to preserve connected components and loops in the mel space.
"""

import torch
import torch.nn as nn
from .mel_fingerprint import TopologicalFingerprint


class TopologicalLoss(nn.Module):
    """Loss that matches Betti numbers between predicted and target spectrograms.

    The key insight: standard pixel-wise L1 doesn't enforce structural
    consistency. Two spectrograms can have identical per-pixel L1 but
    wildly different topology (missing harmonics, extra noise components).
    This loss penalizes Betti number deviations directly.
    """

    def __init__(self, betti_weight: float = 0.1):
        super().__init__()
        self.fingerprint = TopologicalFingerprint(max_dim=1)
        self.betti_weight = betti_weight
        self.pixel_loss = nn.L1Loss()

    def forward(self, pred_mel: torch.Tensor, target_mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_mel: [B, 1, 80, T]
            target_mel: [B, 1, 80, T]

        Returns:
            scalar loss (pixel + topology)
        """
        pixel_term = self.pixel_loss(pred_mel, target_mel)

        # Compute topological fingerprints on CPU (Ripser is NumPy-only)
        with torch.no_grad():
            pred_fp = self.fingerprint(pred_mel.detach().cpu())
            target_fp = self.fingerprint(target_mel.detach().cpu())

        betti_pred = pred_fp["betti"].float().to(pred_mel.device)
        betti_target = target_fp["betti"].float().to(pred_mel.device)

        # L1 on Betti numbers (B0 = components, B1 = loops)
        topology_term = self.pixel_loss(betti_pred, betti_target)

        return pixel_term + self.betti_weight * topology_term
