"""Massive 2D U-Net for mel-spectrogram diffusion — 400M+ parameter variant.

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
from torch.utils.checkpoint import checkpoint


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
    Uses memory-efficient attention to avoid OOM on large spatial dims.
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

        # Memory-efficient attention: compute in chunks to avoid materializing full attn matrix
        # This uses the fact that softmax(q @ k.T) @ v can be computed sequentially
        if H * W > 4096 and self.num_heads >= 8:
            # Use torch.nn.functional.scaled_dot_product_attention if available (PyTorch 2.0+)
            # It uses FlashAttention under the hood
            q = q.transpose(1, 2).contiguous().view(B, H * W, self.num_heads, self.head_dim)
            k = k.transpose(1, 2).contiguous().view(B, H * W, self.num_heads, self.head_dim)
            v = v.transpose(1, 2).contiguous().view(B, H * W, self.num_heads, self.head_dim)
            out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
            out = out.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
            out = out.contiguous().view(B, C, H, W)
        else:
            # Standard attention for smaller spatial dims
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
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


class ResBlock(nn.Module):
    """Residual block: ConvBlock -> ConvBlock + skip."""

    def __init__(self, ch: int, cond_dim: int):
        super().__init__()
        self.block1 = ConvBlock(ch, ch, cond_dim)
        self.block2 = ConvBlock(ch, ch, cond_dim)
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.block1(x, cond)
        h = self.block2(h, cond)
        return x + h


class DownBlock(nn.Module):
    """N ResBlocks + optional attention + downsample."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        n_res: int = 2,
        use_attention: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()
        self.in_conv = ConvBlock(in_ch, out_ch, cond_dim)
        self.res_blocks = nn.ModuleList([
            ResBlock(out_ch, cond_dim) for _ in range(n_res)
        ])
        self.attn = SelfAttention2D(out_ch, num_heads) if use_attention else nn.Identity()
        self.downsample = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x, cond)
        for res in self.res_blocks:
            x = res(x, cond)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """Upsample + N ResBlocks + optional attention."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        cond_dim: int,
        n_res: int = 2,
        use_attention: bool = False,
        num_heads: int = 8,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.in_conv = ConvBlock(in_ch + skip_ch, out_ch, cond_dim)
        self.res_blocks = nn.ModuleList([
            ResBlock(out_ch, cond_dim) for _ in range(n_res)
        ])
        self.attn = SelfAttention2D(out_ch, num_heads) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Align spatial dims with skip (handles odd input sizes)
        if x.shape[2:] != skip.shape[2:]:
            x = torch.nn.functional.interpolate(
                x, size=skip.shape[2:], mode="nearest"
            )
        x = torch.cat([x, skip], dim=1)
        x = self.in_conv(x, cond)
        for res in self.res_blocks:
            x = res(x, cond)
        x = self.attn(x)
        return x


class UNet(nn.Module):
    """Massive 2D U-Net for mel-spectrogram diffusion.

    Args:
        cond_dim: Dimension of the fused conditioning vector (time + text + speaker).
        base_channels: Base channel width. Defaults to 256 for ~400M params.
        channel_mult: Channel multipliers per level.
        n_res_blocks: Number of residual blocks per level.
        attention_levels: Which levels get self-attention.
        use_checkpoint: Enable gradient checkpointing for training memory savings.
    """

    def __init__(
        self,
        cond_dim: int = 512,
        base_channels: int = 256,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        n_res_blocks: int = 3,
        attention_levels: Tuple[int, ...] = (2,),  # NOTE: level 3 has 2048ch, attention OOMs on 8GB
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.cond_dim = cond_dim
        self.use_checkpoint = use_checkpoint

        # Time embedding — larger MLP for bigger model
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(cond_dim),
            nn.Linear(cond_dim, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim),
        )

        chs: List[int] = [base_channels * m for m in channel_mult]

        # Input
        self.input_conv = nn.Conv2d(1, chs[0], kernel_size=3, padding=1, bias=False)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(zip([chs[0]] + chs[:-1], chs)):
            use_attn = i in attention_levels
            heads = 16 if out_ch >= 1024 else 8
            self.down_blocks.append(
                DownBlock(in_ch, out_ch, cond_dim, n_res=n_res_blocks, use_attention=use_attn, num_heads=heads)
            )
            self.down_samples.append(
                nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False)
            )

        # Bottleneck — deep with attention
        mid_ch = chs[-1]
        self.bottleneck = nn.ModuleList([
            ResBlock(mid_ch, cond_dim),
            SelfAttention2D(mid_ch, num_heads=16),
            ResBlock(mid_ch, cond_dim),
            SelfAttention2D(mid_ch, num_heads=16),
            ResBlock(mid_ch, cond_dim),
        ])

        # Decoder
        self.up_blocks = nn.ModuleList()
        rev_chs = list(reversed(chs))
        for i, (in_ch, skip_ch, out_ch) in enumerate(
            zip(rev_chs, rev_chs, rev_chs[1:] + [chs[0]])
        ):
            level_idx = len(chs) - 1 - i
            use_attn = level_idx in attention_levels
            heads = 16 if in_ch >= 1024 else 8
            self.up_blocks.append(
                UpBlock(in_ch, skip_ch, out_ch, cond_dim, n_res=n_res_blocks, use_attention=use_attn, num_heads=heads)
            )

        # Output
        self.output_conv = nn.Conv2d(chs[0], 1, kernel_size=3, padding=1, bias=False)

    def _run_down(self, x, cond, down, ds):
        x = down(x, cond)
        skip = x
        x = ds(x)
        return x, skip

    def _run_up(self, x, skip, cond, up):
        return up(x, skip, cond)

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
            if self.use_checkpoint and self.training:
                x, skip = checkpoint(self._run_down, x, cond, down, ds, use_reentrant=True)
            else:
                x, skip = self._run_down(x, cond, down, ds)
            skips.append(skip)

        for layer in self.bottleneck:
            if isinstance(layer, ResBlock):
                if self.use_checkpoint and self.training:
                    x = checkpoint(layer, x, cond, use_reentrant=True)
                else:
                    x = layer(x, cond)
            else:
                x = layer(x)

        for up in self.up_blocks:
            skip = skips.pop()
            if self.use_checkpoint and self.training:
                x = checkpoint(self._run_up, x, skip, cond, up, use_reentrant=True)
            else:
                x = self._run_up(x, skip, cond, up)

        x = self.output_conv(x)
        return x
