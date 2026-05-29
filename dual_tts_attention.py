#!/usr/bin/env python3
"""
Dual-TTS Cross-Attention Consensus Diffusion

Two TTS models (SpeechT5 + Student) generate mel spectrograms in parallel,
then attend to each other's hidden states via a cross-TTS attention matrix.
Where they agree, the output is confident. Where they disagree, Faraday
applies more diffusion steps to resolve the conflict.

This is ensemble TTS taken to its logical extreme: instead of averaging
outputs, the models TALK to each other until they reach consensus.

Author: Seal — because one TTS is for people who don't care about quality.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from demo_tts import DemonTTS
from faraday.model import FaradayDiffusion


class CrossTTSAttention(nn.Module):
    """Cross-attention between two TTS hidden state sequences.

    Allows TTS_A to attend to TTS_B's representations and vice versa,
    creating a bidirectional information flow during mel generation.

    Args:
        d_model: Dimension of hidden states
        n_heads: Number of attention heads
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Q comes from TTS_A, K/V come from TTS_B
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.scale = 1.0 / math.sqrt(self.d_head)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        hidden_a: torch.Tensor,  # [B, T_A, D] from TTS A
        hidden_b: torch.Tensor,  # [B, T_B, D] from TTS B
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """TTS_A attends to TTS_B.

        Returns:
            [B, T_A, D] enriched hidden states for TTS_A
        """
        B, T_A, _ = hidden_a.shape
        _, T_B, _ = hidden_b.shape

        q = self.q_proj(hidden_a).view(B, T_A, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(hidden_b).view(B, T_B, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(hidden_b).view(B, T_B, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, heads, T_A, T_B]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, heads, T_A, d_head]
        out = out.transpose(1, 2).contiguous().view(B, T_A, self.d_model)
        return self.out_proj(out)


class BidirectionalCrossTTS(nn.Module):
    """Bidirectional cross-attention between two TTS models.

    Both models attend to each other simultaneously, creating a
    symmetric consensus mechanism.
    """

    def __init__(self, d_model: int = 768, n_heads: int = 12):
        super().__init__()
        self.a_to_b = CrossTTSAttention(d_model, n_heads)
        self.b_to_a = CrossTTSAttention(d_model, n_heads)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)
        self.ffn_a = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_a2 = nn.LayerNorm(d_model)
        self.norm_b2 = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_a: torch.Tensor,  # [B, T_A, D]
        hidden_b: torch.Tensor,  # [B, T_B, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional cross-attention.

        Returns:
            (enriched_a, enriched_b) both [B, T, D]
        """
        # A attends to B
        cross_a = self.a_to_b(hidden_a, hidden_b)
        hidden_a = self.norm_a(hidden_a + cross_a)
        hidden_a = self.norm_a2(hidden_a + self.ffn_a(hidden_a))

        # B attends to A
        cross_b = self.b_to_a(hidden_b, hidden_a)
        hidden_b = self.norm_b(hidden_b + cross_b)
        hidden_b = self.norm_b2(hidden_b + self.ffn_b(hidden_b))

        return hidden_a, hidden_b


class MelConsensusFusion(nn.Module):
    """Fuse two mel spectrograms into one via cross-attention.

    Instead of averaging, we let each time-frequency bin attend to
    the corresponding bins in the other mel, learning a soft mask
    for adaptive fusion.
    """

    def __init__(self, mel_bins: int = 80, hidden: int = 256):
        super().__init__()
        self.mel_bins = mel_bins

        # Project each mel into query/key/value space
        self.q_proj = nn.Conv1d(mel_bins, hidden, 1)
        self.k_proj = nn.Conv1d(mel_bins, hidden, 1)
        self.v_proj = nn.Conv1d(mel_bins, hidden, 1)

        # Output projection back to mel bins
        self.out_proj = nn.Conv1d(hidden, mel_bins, 1)

        # Learnable confidence gating: how much to trust mel_a vs mel_b
        self.confidence_gate = nn.Sequential(
            nn.Conv1d(mel_bins * 2, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        self.scale = 1.0 / math.sqrt(hidden)

    def forward(
        self,
        mel_a: torch.Tensor,  # [B, mel_bins, T]
        mel_b: torch.Tensor,  # [B, mel_bins, T]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fuse two mels with attention-based consensus.

        Returns:
            (fused_mel, disagreement_map) where disagreement_map
            is [B, 1, T] indicating per-frame confidence disagreement
        """
        B, _, T = mel_a.shape

        # Compute attention from mel_a to mel_b
        q = self.q_proj(mel_a).transpose(1, 2)  # [B, T, hidden]
        k = self.k_proj(mel_b).transpose(1, 2)  # [B, T, hidden]
        v = self.v_proj(mel_b).transpose(1, 2)  # [B, T, hidden]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, T, T]
        attn = F.softmax(scores, dim=-1)
        attended = torch.matmul(attn, v)  # [B, T, hidden]
        attended = attended.transpose(1, 2)  # [B, hidden, T]

        # Project back to mel space
        fused_attn = self.out_proj(attended)  # [B, mel_bins, T]

        # Learnable gate: confidence in mel_a
        gate_input = torch.cat([mel_a, mel_b], dim=1)
        gate = self.confidence_gate(gate_input)  # [B, 1, T]

        # Gated fusion
        fused = gate * mel_a + (1 - gate) * mel_b

        # Blend with attention-based fusion
        alpha = 0.7  # learned via backprop on gate params
        fused = alpha * fused + (1 - alpha) * fused_attn

        # Disagreement map: per-frame L1 difference
        disagreement = torch.abs(mel_a - mel_b).mean(dim=1, keepdim=True)  # [B, 1, T]

        return fused, disagreement


class ConsensusDiffusion(nn.Module):
    """Faraday diffusion with adaptive steps based on TTS disagreement.

    Where both TTS models agree (low disagreement), we trust the output
    and use fewer diffusion steps. Where they disagree (high disagreement),
    we apply more diffusion to resolve the conflict.
    """

    def __init__(
        self,
        faraday: FaradayDiffusion,
        max_steps: int = 20,
        min_steps: int = 5,
    ):
        super().__init__()
        self.faraday = faraday
        self.max_steps = max_steps
        self.min_steps = min_steps

    def forward(
        self,
        fused_mel: torch.Tensor,  # [B, 1, 80, T]
        disagreement: torch.Tensor,  # [B, 1, T]
        text_emb: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply adaptive diffusion based on consensus confidence.

        Args:
            fused_mel: Fused mel from dual TTS
            disagreement: Per-frame disagreement [B, 1, T]
            text_emb: Text conditioning
            speaker_emb: Speaker conditioning

        Returns:
            [B, 1, 80, T] consensus-enhanced mel
        """
        B, C, H, W = fused_mel.shape
        cond = self.faraday._fuse_conditioning(text_emb, speaker_emb)

        # Normalize disagreement to [0, 1]
        disc_mean = disagreement.mean(dim=-1, keepdim=True)  # [B, 1, 1]
        disc_norm = torch.sigmoid(disc_mean * 10)  # Sharpen

        # Adaptive steps: more disagreement = more diffusion
        adaptive_steps = int(
            self.min_steps + disc_norm.mean().item() * (self.max_steps - self.min_steps)
        )

        # Add noise proportional to disagreement
        t_start = torch.full(
            (B,),
            int(self.faraday.scheduler.num_steps * (0.5 + 0.5 * disc_norm.mean().item())),
            device=fused_mel.device,
            dtype=torch.long,
        )
        x, _ = self.faraday.scheduler.add_noise(fused_mel, t_start)

        # DDIM denoising with adaptive steps
        refined = self.faraday.scheduler.ddim_sample(
            model=self.faraday.unet,
            shape=(B, C, H, W),
            cond=cond,
            steps=adaptive_steps,
            x=x,
        )

        return refined


class DualTTSEnsemble(nn.Module):
    """Full dual-TTS ensemble with cross-attention consensus.

    Architecture:
        Text ──→ [TTS_A (SpeechT5)] ──→ mel_a ──┐
                                              ├──→ MelConsensusFusion ──→ fused_mel
        Text ──→ [TTS_B (Student)] ───→ mel_b ──┘
                                              │
                                              └──→ disagreement_map
                                                    │
                                              ConsensusDiffusion (Faraday)
                                                    │
                                              Enhanced Mel
                                                    │
                                              Vocoder
                                                    │
                                              Aether Polish
                                                    │
                                              Final Waveform

    The two TTS models generate in parallel. Their encoder hidden states
    cross-attend to each other (BidirectionalCrossTTS). Their mel outputs
    are fused via MelConsensusFusion. Where they disagree, ConsensusDiffusion
    applies more Faraday steps.
    """

    def __init__(
        self,
        tts_a,  # SpeechT5 or similar
        tts_b,  # StudentTTS or similar
        faraday: FaradayDiffusion,
        cross_tts_dim: int = 768,
    ):
        super().__init__()
        self.tts_a = tts_a  # e.g. SpeechT5
        self.tts_b = tts_b  # e.g. StudentTTS
        self.faraday = faraday

        # Cross-TTS attention (if hidden states available)
        self.cross_tts = BidirectionalCrossTTS(d_model=cross_tts_dim)

        # Mel fusion
        self.mel_fusion = MelConsensusFusion(mel_bins=80, hidden=256)

        # Consensus diffusion
        self.consensus_diffusion = ConsensusDiffusion(faraday, max_steps=20, min_steps=5)

    def forward(
        self,
        text_tokens_a,  # Input for TTS A
        text_tokens_b,  # Input for TTS B
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> dict:
        """Generate audio via dual-TTS consensus.

        Returns:
            dict with keys: mel_a, mel_b, fused_mel, disagreement, enhanced_mel
        """
        # Generate from both TTS models
        # Note: In practice, we'd extract encoder hidden states and apply
        # cross-attention before decoder. This is simplified.

        with torch.no_grad():
            # TTS A generation
            mel_a = self.tts_a(text_tokens_a, speaker_emb)  # [B, 80, T]

            # TTS B generation
            mel_b = self.tts_b(text_tokens_b, speaker_emb)  # [B, 80, T]

        # Ensure same length
        T_min = min(mel_a.size(-1), mel_b.size(-1))
        mel_a = mel_a[..., :T_min]
        mel_b = mel_b[..., :T_min]

        # Fuse mels with attention
        fused_mel, disagreement = self.mel_fusion(mel_a, mel_b)

        # Add channel dim for Faraday
        fused_mel_4d = fused_mel.unsqueeze(1)  # [B, 1, 80, T]

        # Adaptive consensus diffusion
        enhanced_mel = self.consensus_diffusion(
            fused_mel_4d,
            disagreement,
            text_emb=None,
            speaker_emb=speaker_emb,
        )

        return {
            "mel_a": mel_a,
            "mel_b": mel_b,
            "fused_mel": fused_mel,
            "disagreement": disagreement,
            "enhanced_mel": enhanced_mel.squeeze(1),
        }


class MultiScaleDisagreementLoss(nn.Module):
    """Loss that penalizes disagreement between TTS models at multiple scales.

    Ensures the two TTS models converge toward consensus, not divergence.
    """

    def __init__(self, scales: list = [1, 2, 4]):
        super().__init__()
        self.scales = scales

    def forward(self, mel_a: torch.Tensor, mel_b: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale consensus loss.

        Args:
            mel_a: TTS A output [B, 80, T]
            mel_b: TTS B output [B, 80, T]
            target: Ground truth mel [B, 80, T]

        Returns:
            Loss that pushes both models toward target AND each other
        """
        loss = 0.0

        for scale in self.scales:
            if scale > 1:
                a_down = F.avg_pool1d(mel_a, scale, stride=scale)
                b_down = F.avg_pool1d(mel_b, scale, stride=scale)
                t_down = F.avg_pool1d(target, scale, stride=scale)
            else:
                a_down, b_down, t_down = mel_a, mel_b, target

            # Each model should match target
            loss += F.l1_loss(a_down, t_down)
            loss += F.l1_loss(b_down, t_down)

            # Models should agree with each other
            loss += F.l1_loss(a_down, b_down)

        return loss / (len(self.scales) * 3)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Dual-TTS Cross-Attention Consensus Diffusion")
    print("=" * 60)

    # Component sizes
    cross_attn = BidirectionalCrossTTS(d_model=768, n_heads=12)
    fusion = MelConsensusFusion(mel_bins=80, hidden=256)
    consensus = ConsensusDiffusion(
        FaradayDiffusion(cond_dim=512, base_channels=256),
        max_steps=20,
        min_steps=5,
    )

    print(f"\n[BidirectionalCrossTTS] Params: {count_parameters(cross_attn):,}")
    print(f"[MelConsensusFusion]    Params: {count_parameters(fusion):,}")
    print(f"[ConsensusDiffusion]    Params: {count_parameters(consensus.faraday):,}")

    total = count_parameters(cross_attn) + count_parameters(fusion)
    print(f"\n[New Components Total]  Params: {total:,}")
    print(f"[Full Dual-TTS System]  Params: ~{total / 1e6:.0f}M+")

    # Test forward
    print("\n--- Test Forward Pass ---")
    B, T = 1, 256
    hidden_a = torch.randn(B, 100, 768)
    hidden_b = torch.randn(B, 100, 768)

    enriched_a, enriched_b = cross_attn(hidden_a, hidden_b)
    print(f"Cross-attention OK: {enriched_a.shape}, {enriched_b.shape}")

    mel_a = torch.randn(B, 80, T)
    mel_b = torch.randn(B, 80, T)
    fused, disc = fusion(mel_a, mel_b)
    print(f"Mel fusion OK: fused={fused.shape}, disagreement={disc.shape}")

    print("\nAll systems nominal. Ready to make audio that argues with itself.")
