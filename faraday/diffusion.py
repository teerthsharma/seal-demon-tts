"""DDPM / DDIM scheduler and training/sampling utilities."""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionScheduler(nn.Module):
    """Cosine beta schedule DDPM scheduler with DDIM sampling support.

    Pre-computes alphas and alpha_bars for 1000 steps.
    """

    def __init__(self, num_steps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        super().__init__()
        self.num_steps = num_steps

        # Cosine beta schedule (improved from Nichol & Dhariwal, 2021)
        steps = torch.arange(num_steps + 1, dtype=torch.float64)
        s = 0.008
        f_t = torch.cos(((steps / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alpha_bar = f_t / f_t[0]
        betas = torch.clip(1.0 - (alpha_bar[1:] / alpha_bar[:-1]), 0.0001, 0.9999)
        alphas = 1.0 - betas

        # Register buffers so they move with .to(device)
        self.register_buffer("betas", betas.float())
        self.register_buffer("alphas", alphas.float())
        self.register_buffer("alpha_bar", alpha_bar[1:].float())
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar[1:]).float())
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar[1:]).float()
        )
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas).float())
        self.register_buffer(
            "posterior_variance",
            (betas * (1.0 - alpha_bar[:-1]) / (1.0 - alpha_bar[1:])).float(),
        )

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Q-sample: add noise to clean data x0 at timestep t.

        Args:
            x0:     [B, C, H, W] clean mel
            t:      [B] int64 timesteps
            noise:  optional noise tensor; sampled if None

        Returns:
            (noisy_x, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t][:, None, None, None]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t][:, None, None, None]
        noisy = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return noisy, noise

    def training_loss(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE loss between predicted and true noise.

        Args:
            model:      noise-prediction U-Net (or wrapper)
            x0:         [B, 1, 80, T] clean mel
            cond:       [B, cond_dim] fused conditioning

        Returns:
            scalar loss tensor
        """
        B = x0.shape[0]
        device = x0.device
        t = torch.randint(0, self.num_steps, (B,), device=device).long()
        noise = torch.randn_like(x0)
        noisy, _ = self.add_noise(x0, t, noise=noise)
        pred_noise = model(noisy, t, cond)
        loss = F.mse_loss(pred_noise, noise)
        return loss

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        cond: torch.Tensor,
        steps: int = 10,
        eta: float = 0.0,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """DDIM deterministic sampling loop.

        Args:
            model:      noise-prediction U-Net (or wrapper)
            shape:      output tensor shape, e.g. (B, 1, 80, T)
            cond:       [B, cond_dim] fused conditioning
            steps:      number of DDIM steps (default 10)
            eta:        stochasticity (0 = deterministic)
            x:          optional initial latent; random noise if None

        Returns:
            [B, 1, 80, T] denoised mel
        """
        device = cond.device
        B = shape[0]
        if x is None:
            x = torch.randn(shape, device=device)

        # Create sub-sequence of timesteps
        timesteps = torch.linspace(self.num_steps - 1, 0, steps + 1, device=device).long()

        for i in range(steps):
            t_cur = timesteps[i].unsqueeze(0).expand(B)
            t_next = timesteps[i + 1].unsqueeze(0).expand(B)

            # Predict noise
            pred_noise = model(x, t_cur, cond)

            alpha_bar_t = self.alpha_bar[t_cur][:, None, None, None]
            alpha_bar_next = self.alpha_bar[t_next][:, None, None, None]
            alpha_bar_next = torch.where(
                t_next[:, None, None, None] < 0,
                torch.ones_like(alpha_bar_next),
                alpha_bar_next,
            )

            # Predict x0
            pred_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * pred_noise) / torch.sqrt(
                alpha_bar_t
            )
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

            # DDIM direction
            sigma_t = eta * torch.sqrt(
                (1.0 - alpha_bar_next) / (1.0 - alpha_bar_t)
                * (1.0 - alpha_bar_t / alpha_bar_next)
            )
            noise_dir = torch.sqrt(1.0 - alpha_bar_next - sigma_t**2) * pred_noise

            x = torch.sqrt(alpha_bar_next) * pred_x0 + noise_dir
            if eta > 0 and i < steps - 1:
                x = x + sigma_t * torch.randn_like(x)

        return x
