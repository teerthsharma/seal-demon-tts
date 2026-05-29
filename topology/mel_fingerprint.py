"""Compute topological fingerprints of mel spectrograms using persistent homology.

Adapted from faraday-main/barcode.py for audio use. Converts mel spectrograms
into point clouds, then computes Betti numbers via Ripser.
"""

import numpy as np
import torch
from typing import Tuple, List

try:
    from ripser import ripser
    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


def mel_to_pointcloud(mel: np.ndarray, threshold_db: float = -80.0, max_points: int = 200) -> np.ndarray:
    """Convert mel spectrogram to weighted point cloud for Ripser.

    Args:
        mel: [80, T] log-mel spectrogram
        threshold_db: minimum dB value to include as a point
        max_points: maximum points to pass to Ripser (speed limit)

    Returns:
        [N, 3] point cloud where each point is (freq_bin, time_frame, magnitude)
    """
    mel_norm = mel - mel.min()
    if mel_norm.max() > 0:
        mel_norm = mel_norm / mel_norm.max()

    # Adaptive threshold based on dB floor
    threshold = (threshold_db / -80.0) * 0.1 + 0.01
    mask = mel_norm > threshold

    points = []
    freqs, times = np.where(mask)
    for f, t in zip(freqs, times):
        mag = mel_norm[f, t]
        points.append([f / 80.0, t / max(1, mel.shape[1]), mag])

    if len(points) < 10:
        # Fallback: uniform sampling to ensure Ripser has enough points
        f_idx = np.linspace(0, 79, 20).astype(int)
        t_idx = np.linspace(0, mel.shape[1] - 1, 20).astype(int)
        for f in f_idx:
            for t in t_idx:
                points.append([
                    f / 80.0,
                    t / max(1, mel.shape[1]),
                    mel_norm[f, t] + 0.01,
                ])

    # Speed limit: Ripser chokes on >200 points
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points).astype(int)
        points = [points[i] for i in indices]

    return np.array(points, dtype=np.float32)


def compute_betti_numbers(
    pointcloud: np.ndarray, max_dim: int = 1
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Compute Betti-0 and Betti-1 numbers from point cloud via Ripser.

    Args:
        pointcloud: [N, D] array of points
        max_dim: maximum homology dimension (1 = Betti-0 and Betti-1)

    Returns:
        (diagrams, betti_numbers) where betti_numbers is [B0, B1]
    """
    if not RIPSER_AVAILABLE or len(pointcloud) < 5:
        return (
            [np.array([]), np.array([])],
            np.array([min(10, len(pointcloud)), 0]),
        )

    result = ripser(pointcloud, maxdim=max_dim)
    diagrams = result["dgms"]

    betti = []
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            betti.append(0)
            continue

        lifetimes = dgm[:, 1] - dgm[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]

        if len(lifetimes) > 0 and lifetimes.max() > 0:
            thresh = lifetimes.max() * 0.1
            betti.append(int((lifetimes > thresh).sum()))
        else:
            betti.append(0)

    return diagrams, np.array(betti, dtype=np.int32)


class TopologicalFingerprint:
    """Compute topological fingerprints for mel spectrograms."""

    def __init__(self, max_dim: int = 1):
        self.max_dim = max_dim

    def __call__(self, mel: torch.Tensor) -> dict:
        """Compute fingerprint for a mel spectrogram.

        Args:
            mel: [B, 1, 80, T] batched or [80, T] single

        Returns:
            dict with 'betti' tensor and 'diagrams' list
        """
        if mel.dim() == 4:
            mel = mel.squeeze(1)
            results = []
            for i in range(mel.shape[0]):
                results.append(self._process_single(mel[i]))
            return {
                "betti": torch.stack([r["betti"] for r in results]),
                "diagrams": [r["diagrams"] for r in results],
            }
        return self._process_single(mel)

    def _process_single(self, mel: torch.Tensor) -> dict:
        mel_np = mel.detach().cpu().numpy()
        pc = mel_to_pointcloud(mel_np)
        diagrams, betti = compute_betti_numbers(pc, self.max_dim)
        return {
            "betti": torch.from_numpy(betti),
            "diagrams": diagrams,
        }
