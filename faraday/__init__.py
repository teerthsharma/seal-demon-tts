"""Faraday Diffusion: 2D mel-spectrogram diffusion enhancer.

A lightweight diffusion U-Net operating on mel spectrograms, conditioned on
text and speaker embeddings. Supports DDIM sampling with as few as 10 steps
and ONNX export for deployment.
"""

from .diffusion import DiffusionScheduler
from .model import FaradayDiffusion
from .unet import UNet, count_parameters

__all__ = [
    "DiffusionScheduler",
    "FaradayDiffusion",
    "UNet",
    "count_parameters",
]
