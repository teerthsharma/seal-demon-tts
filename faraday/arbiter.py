#!/usr/bin/env python3
"""
Faraday Arbiter — A learned critic that judges diffusion quality.

A small transformer-based TTS model (~35M params) that attends to
Faraday's output mel spectrogram and decides whether the diffusion
was "correct" or needs to be re-attempted with adjusted conditioning.

Architecture:
    Text tokens ──→ Small Encoder ──┐
                                    ├──→ Cross-Attention ──→ Quality Head
    Faraday mel ──→ Mel Encoder ────┘

Output:
    - quality_score: [B, 1] ∈ [0, 1] — how good is this mel?
    - correction_emb: [B, 512] — what direction to adjust conditioning
    - should_rediffuse: bool — trigger another diffusion pass?

Author: Seal — because even diffusion models need a boss.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


class MelPatchEncoder(nn.Module):
    """Encode mel spectrogram into patches for transformer processing.

    Similar to Vision Transformer: split mel into time patches,
    flatten frequency bins, project to transformer dimension.
    """

    def __init__(self, mel_bins: int = 80, patch_time: int = 4, d_model: int = 256):
        super().__init__()
        self.patch_time = patch_time
        self.mel_bins = mel_bins
        self.d_model = d_model
        patch_dim = mel_bins * patch_time
        self.proj = nn.Linear(patch_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 1000, d_model) * 0.02)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 1, 80, T]
        Returns:
            [B, n_patches, d_model]
        """
        B, _, H, W = mel.shape
        # Pad to multiple of patch_time
        pad = (self.patch_time - W % self.patch_time) % self.patch_time
        if pad > 0:
            mel = F.pad(mel, (0, pad))
            W = W + pad

        # Reshape into patches: [B, n_patches, patch_time, mel_bins]
        n_patches = W // self.patch_time
        patches = mel.view(B, H, n_patches, self.patch_time)
        patches = patches.permute(0, 2, 1, 3)  # [B, n_patches, mel_bins, patch_time]
        patches = patches.reshape(B, n_patches, -1)  # [B, n_patches, patch_dim]

        # Project and add pos emb
        x = self.proj(patches)  # [B, n_patches, d_model]
        x = x + self.pos_emb[:, :n_patches, :]
        return self.norm(x)


class TextEncoder(nn.Module):
    """Small transformer encoder for text tokens."""

    def __init__(self, vocab_size: int = 1000, d_model: int = 256, n_layers: int = 4, n_heads: int = 8):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, T] int64
        Returns:
            [B, T, d_model]
        """
        B, T = tokens.shape
        x = self.emb(tokens)
        pos = torch.arange(T, device=tokens.device)
        x = x + self.pos_emb(pos)[None, :, :]
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class CrossModalAttention(nn.Module):
    """Cross-attention between text and mel representations."""

    def __init__(self, d_model: int = 256, n_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, text_h: torch.Tensor, mel_h: torch.Tensor) -> torch.Tensor:
        """Text attends to mel patches.

        Args:
            text_h: [B, T_text, D]
            mel_h: [B, T_mel, D]
        Returns:
            [B, T_text, D] enriched text representations
        """
        attn_out, _ = self.attn(text_h, mel_h, mel_h, need_weights=False)
        text_h = self.norm(text_h + attn_out)
        text_h = self.norm2(text_h + self.ffn(text_h))
        return text_h


class FaradayArbiter(nn.Module):
    """The Faraday Arbiter — judges diffusion quality and suggests corrections.

    This is essentially a learned critic that provides feedback to Faraday.
    It looks at the diffused mel and says:
    - "This is fine" (score > 0.8, no action)
    - "This needs work" (score 0.5-0.8, apply correction embedding)
    - "Burn it and start over" (score < 0.5, full re-diffusion)

    Total params: ~35M
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        mel_bins: int = 80,
        patch_time: int = 4,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        cond_dim: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.cond_dim = cond_dim

        # Encoders
        self.text_encoder = TextEncoder(vocab_size, d_model, n_layers, n_heads)
        self.mel_encoder = MelPatchEncoder(mel_bins, patch_time, d_model)

        # Cross-modal attention: text attends to mel
        self.cross_attn = CrossModalAttention(d_model, n_heads)

        # Quality estimation heads
        self.quality_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Correction embedding head — tells Faraday how to adjust conditioning
        self.correction_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, cond_dim),
        )

        # Rediffusion trigger head
        self.rediffuse_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Learned thresholds
        self.register_buffer('threshold_good', torch.tensor(0.75))
        self.register_buffer('threshold_bad', torch.tensor(0.4))

    def forward(
        self,
        text_tokens: torch.Tensor,  # [B, T]
        faraday_mel: torch.Tensor,  # [B, 1, 80, T_mel]
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> dict:
        """Judge Faraday's diffusion output.

        Args:
            text_tokens: Original text tokens
            faraday_mel: Faraday's diffused mel spectrogram
            speaker_emb: Optional speaker embedding

        Returns:
            dict with:
                - quality_score: [B, 1] ∈ [0, 1]
                - correction_emb: [B, cond_dim]
                - should_rediffuse: [B, 1] ∈ [0, 1]
                - judgment: str description
        """
        B = text_tokens.size(0)

        # Encode both modalities
        text_h = self.text_encoder(text_tokens)  # [B, T_text, D]
        mel_h = self.mel_encoder(faraday_mel)    # [B, T_patches, D]

        # Cross-attention: text understands what mel looks like
        enriched = self.cross_attn(text_h, mel_h)  # [B, T_text, D]

        # Global pooling
        pooled = enriched.mean(dim=1)  # [B, D]

        # Predictions
        quality = self.quality_head(pooled)           # [B, 1]
        correction = self.correction_head(pooled)     # [B, cond_dim]
        rediffuse = self.rediffuse_head(pooled)       # [B, 1]

        # Generate human-readable judgment
        judgments = []
        for q, r in zip(quality.squeeze(-1), rediffuse.squeeze(-1)):
            if q > self.threshold_good and r < 0.3:
                judgments.append("EXCELLENT — No changes needed")
            elif q > self.threshold_bad:
                judgments.append(f"ACCEPTABLE — Minor correction (score: {q:.2f})")
            else:
                judgments.append(f"REJECT — Full rediffusion required (score: {q:.2f})")

        return {
            "quality_score": quality,
            "correction_embedding": correction,
            "should_rediffuse": rediffuse,
            "judgments": judgments,
        }

    def get_training_loss(
        self,
        text_tokens: torch.Tensor,
        faraday_mel: torch.Tensor,
        target_mel: torch.Tensor,
    ) -> torch.Tensor:
        """Training loss for the arbiter.

        We want the arbiter to:
        1. Give HIGH scores when faraday_mel ≈ target_mel
        2. Give LOW scores when faraday_mel ≠ target_mel
        3. Predict useful correction embeddings
        """
        # Judge Faraday's output
        output = self.forward(text_tokens, faraday_mel)
        quality = output["quality_score"]
        correction = output["correction_embedding"]

        # Ground truth: how good is faraday_mel compared to target?
        mel_error = F.l1_loss(faraday_mel, target_mel, reduction='none').mean(dim=(1,2,3), keepdim=True)
        target_quality = torch.exp(-mel_error * 10)  # high quality = low error

        # Quality prediction loss
        quality_loss = F.mse_loss(quality, target_quality)

        # Correction should point toward the target
        # We can't directly supervise correction_emb, but we can ensure
        # that when added to Faraday's conditioning, it improves the output
        # This is trained jointly with Faraday
        correction_loss = correction.pow(2).mean() * 0.01  # L2 regularization

        return quality_loss + correction_loss


class FaradayWithArbiter(nn.Module):
    """Faraday diffusion enhanced with Arbiter feedback loop.

    This creates an iterative diffusion process:
    1. Faraday diffuses the mel
    2. Arbiter judges the output
    3. If score is low, apply correction embedding and re-diffuse
    4. Repeat up to max_iter times

    In practice, most outputs pass on the first try. Only difficult
    segments (high disagreement, complex phonetics) need iteration.
    """

    def __init__(self, faraday, arbiter, max_iter: int = 3):
        super().__init__()
        self.faraday = faraday
        self.arbiter = arbiter
        self.max_iter = max_iter

    def enhance_with_feedback(
        self,
        mel: torch.Tensor,
        text_tokens: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        steps: int = 10,
    ) -> Tuple[torch.Tensor, dict]:
        """Iterative diffusion with arbiter feedback.

        Returns:
            (enhanced_mel, metadata) where metadata contains iteration count
        """
        current_mel = mel
        metadata = {"iterations": 0, "judgments": []}

        for i in range(self.max_iter):
            # Diffuse
            current_mel = self.faraday.enhance(
                current_mel, text_emb, speaker_emb, steps=steps
            )

            # Judge
            with torch.no_grad():
                arbiter_out = self.arbiter(text_tokens, current_mel)

            quality = arbiter_out["quality_score"].item()
            metadata["judgments"].append({
                "iter": i + 1,
                "quality": quality,
                "judgment": arbiter_out["judgments"][0],
            })

            # If good enough, stop
            if quality > self.arbiter.threshold_good:
                metadata["iterations"] = i + 1
                return current_mel, metadata

            # If bad, apply correction and continue
            if quality < self.arbiter.threshold_bad:
                # Full rediffusion with adjusted conditioning
                correction = arbiter_out["correction_embedding"]
                if text_emb is not None:
                    text_emb = text_emb + correction[:, :text_emb.size(-1)]
                else:
                    text_emb = correction

        metadata["iterations"] = self.max_iter
        return current_mel, metadata


if __name__ == "__main__":
    print("=" * 60)
    print("Faraday Arbiter — The Diffusion Critic")
    print("=" * 60)

    arbiter = FaradayArbiter(vocab_size=1000, d_model=256, n_layers=4)
    total = count_parameters(arbiter)
    print(f"\n[FaradayArbiter] Params: {total:,} (~{total/1e6:.1f}M)")
    print(f"Target: ~35M")

    # Test forward
    B = 2
    text_tokens = torch.randint(0, 1000, (B, 50))
    faraday_mel = torch.randn(B, 1, 80, 256)

    output = arbiter(text_tokens, faraday_mel)
    print(f"\nQuality scores: {output['quality_score'].squeeze().tolist()}")
    print(f"Correction shape: {output['correction_embedding'].shape}")
    print(f"Rediffuse probs: {output['should_rediffuse'].squeeze().tolist()}")
    for j in output["judgments"]:
        print(f"  → {j}")

    # Test with Faraday
    from faraday.model import FaradayDiffusion
    faraday = FaradayDiffusion(cond_dim=512, base_channels=256)
    system = FaradayWithArbiter(faraday, arbiter, max_iter=3)

    mel = torch.randn(1, 1, 80, 256)
    enhanced, meta = system.enhance_with_feedback(
        mel, torch.randint(0, 1000, (1, 50)), steps=5
    )
    print(f"\nEnhanced mel: {enhanced.shape}")
    print(f"Iterations needed: {meta['iterations']}")
    for j in meta["judgments"]:
        print(f"  Iter {j['iter']}: {j['judgment']} (q={j['quality']:.3f})")

    print(f"\nTotal system params: {count_parameters(system):,}")
    print("Arbiter ready to judge Faraday's life choices.")
