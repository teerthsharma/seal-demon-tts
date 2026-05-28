"""Lightweight 2D U-Net for mel-spectrogram diffusion.

Input:  mel          [B, 1, 80, T]
        timestep     [B]
        cond         [B, cond_dim]   (time + text + speaker fused)
Output: residual     [B, 1, 80, T]
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal time embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        return embeddings


class FiLM(nn.Module):
    """Feature-wise Linear Modulation.
    Projects a conditional vector to per-channel scale and shift.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, channels)
        self.shift = nn.Linear(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], cond: [B, cond_dim]
        scale = self.scale(cond)[:, :, None, None]
        shift = self.shift(cond)[:, :, None, None]
        return x * (1.0 + scale) + shift


class SelfAttention2D(nn.Module):
    """Multi-head self-attention for 2D feature maps.
    Operates on flattened spatial dimensions.
    """

    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # [B, 3*C, H, W]
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to [B, heads, HW, head_dim]
        q = q.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, self.head_dim, H * W).transpose(2, 3)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, HW, head_dim]

        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.proj(out)
        return x + out


class ConvBlock(nn.Module):
    """Conv2d -> GroupNorm -> SiLU -> FiLM."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm = nn.GroupNorm(min(32, out_ch), out_ch)
        self.film = FiLM(cond_dim, out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.film(x, cond)
        return x


class DownBlock(nn.Module):
    """Two ConvBlocks + optional attention."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        use_attention: bool = False,
    ):
        super().__init__()
        self.block1 = ConvBlock(in_ch, out_ch, cond_dim)
        self.block2 = ConvBlock(out_ch, out_ch, cond_dim)
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Upsample + two ConvBlocks."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        cond_dim: int,
        use_attention: bool = False,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.block1 = ConvBlock(in_ch + skip_ch, out_ch, cond_dim)
        self.block2 = ConvBlock(out_ch, out_ch, cond_dim)
        self.attn = SelfAttention2D(out_ch) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Align spatial dims with skip (handles odd input sizes)
        if x.shape[2:] != skip.shape[2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[2:], mode="nearest"
            )
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, cond)
        x = self.block2(x, cond)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    """Lightweight 2D U-Net for mel-spectrogram diffusion.

    Args:
        cond_dim: Dimension of the fused conditioning vector (time + text + speaker).
        base_channels: Base channel width. Defaults to 64 for ~20M params.
    """

    def __init__(self, cond_dim: int = 128, base_channels: int = 64):
        super().__init__()
        self.cond_dim = cond_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(cond_dim),
            nn.Linear(cond_dim, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

        # Text / speaker projections (applied externally in model.py, but we
        # accept a pre-fused cond vector here).

        chs: List[int] = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
        ]

        # Input
        self.input_conv = nn.Conv2d(1, chs[0], kernel_size=3, padding=1, bias=False)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(zip([chs[0]] + chs[:-1], chs)):
            use_attn = False  # attention only in bottleneck for memory
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, cond_dim, use_attention=use_attn)
            )
            self.down_samples.append(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False)
            )

        # Bottleneck
        mid_ch = chs[-1]
        self.bottleneck1 = ConvBlock(mid_ch, mid_ch, cond_dim)
        self.bottleneck_attn = SelfAttention2D(mid_ch)
        self.bottleneck2 = ConvBlock(mid_ch, mid_ch, cond_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        rev_chs = list(reversed(chs))
        for i, (in_ch, skip_ch, out_ch) in enumerate(
            zip(rev_chs, rev_chs, rev_chs[1:] + [chs[0]])
        ):
            use_attn = False
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, cond_dim, use_attention=use_attn)
            )

        # Output
        self.output_conv = nn.Conv2d(chs[0], 1, kernel_size=3, padding=1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x:          [B, 1, 80, T]
            timestep:   [B]
            cond:       [B, cond_dim]  (text + speaker conditioning)

        Returns:
            [B, 1, 80, T] residual
        """
        # Fuse time into conditioning
        t_emb = self.time_mlp(timestep)  # [B, cond_dim]
        cond = cond + t_emb

        x = self.input_conv(x)

        skips: List[torch.Tensor] = []
        for down, ds in zip(self.down_blocks, self.down_samples):
            x = down(x, cond)
            skips.append(x)
            x = ds(x)

        x = self.bottleneck1(x, cond)
        x = self.bottleneck_attn(x)
        x = self.bottleneck2(x, cond)

        for up in self.up_blocks:
            skip = skips.pop()
            x = up(x, skip, cond)

        x = self.output_conv(x)
        return x
