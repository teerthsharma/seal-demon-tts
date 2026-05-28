"""180M-parameter TTS Student transformer with RoPE and speaker cross-attention."""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class RoPE(nn.Module):
    """Rotary positional embeddings."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T, D]
        T = x.size(2)
        t = torch.arange(T, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        return apply_rotary_pos_emb(x, cos_emb, sin_emb)


def apply_rotary_pos_emb(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    y1 = x1 * cos[..., ::2] - x2 * sin[..., ::2]
    y2 = x1 * sin[..., ::2] + x2 * cos[..., ::2]
    return torch.stack([y1, y2], dim=-1).flatten(-2)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        _, S, _ = cond.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        kv = self.kv_proj(cond).view(B, S, 2, self.n_heads, self.d_head)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RoPE(d_model // n_heads)

    def forward(self, x: torch.Tensor, speaker_emb: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with RoPE
        B, T, _ = x.shape
        q = x.view(B, T, -1)  # placeholder for rope application inside MHA
        # Standard MHA doesn't expose heads, so we apply RoPE manually by splitting
        # Simplification: use standard MHA for now, RoPE can be injected via custom MHA if needed.
        attn_out, _ = self.self_attn(q, q, q, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention to speaker embedding
        cross_out = self.cross_attn(x, speaker_emb)
        x = self.norm2(x + self.dropout(cross_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class StudentTTS(nn.Module):
    """180M param transformer TTS backbone."""

    def __init__(
        self,
        vocab_size: int = 10_000,
        d_model: int = 1024,
        n_layers: int = 10,
        n_heads: int = 16,
        d_ff: int = 4096,
        mel_bins: int = 80,
        max_seq_len: int = 2048,
        speaker_dim: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.speaker_proj = nn.Linear(speaker_dim, d_model)

        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.mel_proj = nn.Linear(d_model, mel_bins)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
        mel_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_tokens: [B, T] int64
            speaker_embedding: [B, speaker_dim] or [B, 1, speaker_dim]
            mel_target: [B, mel_bins, T] optional for loss computation
        Returns:
            mel: [B, mel_bins, T]
        """
        B, T = text_tokens.shape
        x = self.token_emb(text_tokens)
        pos = torch.arange(T, device=x.device)
        x = x + self.pos_emb(pos)[None, :, :]
        x = self.dropout(x)

        if speaker_embedding.dim() == 2:
            speaker_embedding = speaker_embedding[:, None, :]  # [B, 1, D]
        speaker_emb = self.speaker_proj(speaker_embedding)  # [B, 1, d_model]

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        mask = mask.masked_fill(mask, float('-inf'))

        for layer in self.layers:
            x = layer(x, speaker_emb, mask)

        mel = self.mel_proj(x).transpose(1, 2)  # [B, mel_bins, T]

        if mel_target is not None:
            loss = F.l1_loss(mel, mel_target)
            return mel, loss
        return mel


if __name__ == "__main__":
    model = StudentTTS()
    total = count_parameters(model)
    print(f"[Student] Total parameters: {total:,} (~{total/1e6:.1f}M)")

    tok = torch.randint(0, 10_000, (2, 128))
    spk = torch.randn(2, 192)
    mel, loss = model(tok, spk, mel_target=torch.randn(2, 80, 128))
    print(f"Output shape: {mel.shape}, loss: {loss.item():.4f}")
