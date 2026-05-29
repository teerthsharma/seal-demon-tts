"""Topology module: persistent homology for mel spectrograms.

Computes topological fingerprints (Betti numbers, persistence diagrams)
using Ripser. Used to guide diffusion and as a training loss component.
"""

from .mel_fingerprint import TopologicalFingerprint, compute_betti_numbers, mel_to_pointcloud
from .barcode_loss import TopologicalLoss

__all__ = [
    "TopologicalFingerprint",
    "compute_betti_numbers",
    "mel_to_pointcloud",
    "TopologicalLoss",
]
