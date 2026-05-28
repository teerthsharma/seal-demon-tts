"""Combined diffusion wrapper: U-Net + scheduler."""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .diffusion import DiffusionScheduler
from .unet import UNet, count_parameters


class FaradayDiffusion(nn.Module):
    """Mel-spectrogram diffusion enhancer.

    Holds a lightweight 2D U-Net and a DDPM/DDIM scheduler.
    """

    def __init__(
        self,
        text_dim: int = 512,
        speaker_dim: int = 256,
        cond_dim: int = 128,
        base_channels: int = 64,
        num_steps: int = 1000,
    ):
        super().__init__()
        self.text_dim = text_dim
        self.speaker_dim = speaker_dim
        self.cond_dim = cond_dim

        self.text_proj = nn.Linear(text_dim, cond_dim)
        self.speaker_proj = nn.Linear(speaker_dim, cond_dim)
        self.cond_act = nn.SiLU()

        self.unet = UNet(cond_dim=cond_dim, base_channels=base_channels)
        self.scheduler = DiffusionScheduler(num_steps=num_steps)

    def _fuse_conditioning(
        self,
        text_emb: Optional[torch.Tensor],
        speaker_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse text and speaker embeddings into a single cond vector."""
        B = text_emb.shape[0] if text_emb is not None else speaker_emb.shape[0]
        device = text_emb.device if text_emb is not None else speaker_emb.device
        cond = torch.zeros(B, self.cond_dim, device=device)
        if text_emb is not None:
            if text_emb.dim() == 3:
                text_emb = text_emb.mean(dim=1)  # pool seq dim -> [B, text_dim]
            cond = cond + self.text_proj(text_emb)
        if speaker_emb is not None:
            if speaker_emb.dim() == 3:
                speaker_emb = speaker_emb.mean(dim=1)
            cond = cond + self.speaker_proj(speaker_emb)
        return self.cond_act(cond)

    def forward(
        self,
        noisy_mel: torch.Tensor,
        t: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise from a noisy mel.

        Args:
            noisy_mel:  [B, 1, 80, T]
            t:          [B] timestep indices
            text_emb:   [B, text_dim]
            speaker_emb:[B, speaker_dim]

        Returns:
            [B, 1, 80, T] predicted noise
        """
        cond = self._fuse_conditioning(text_emb, speaker_emb)
        return self.unet(noisy_mel, t, cond)

    def enhance(
        self,
        mel: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
        steps: int = 10,
    ) -> torch.Tensor:
        """Denoise / enhance a mel spectrogram using DDIM.

        The input mel is noised to the highest timestep and then denoised
        back in `steps` DDIM iterations, conditioned on text and speaker.

        Args:
            mel:        [B, 1, 80, T]  reference mel (used for shape and noising)
            text_emb:   [B, text_dim]
            speaker_emb:[B, speaker_dim]
            steps:      number of DDIM steps

        Returns:
            [B, 1, 80, T] refined mel
        """
        B, C, H, W = mel.shape
        cond = self._fuse_conditioning(text_emb, speaker_emb)
        t_start = torch.full(
            (B,), self.scheduler.num_steps - 1, device=mel.device, dtype=torch.long
        )
        x, _ = self.scheduler.add_noise(mel, t_start)
        return self.scheduler.ddim_sample(
            model=self.unet,
            shape=(B, C, H, W),
            cond=cond,
            steps=steps,
            x=x,
        )

    def training_loss(
        self,
        mel: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        speaker_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convenience wrapper around scheduler.training_loss."""
        cond = self._fuse_conditioning(text_emb, speaker_emb)
        return self.scheduler.training_loss(self.unet, mel, cond)

    def export_onnx(self, path: str, mel_length: int = 256) -> None:
        """Export the U-Net to ONNX for inference.

        Args:
            path:       destination ONNX file path
            mel_length: time dimension T (default 256)
        """
        self.eval()
        dummy_mel = torch.randn(1, 1, 80, mel_length)
        dummy_t = torch.tensor([0], dtype=torch.long)
        dummy_cond = torch.randn(1, self.cond_dim)

        torch.onnx.export(
            self.unet,
            (dummy_mel, dummy_t, dummy_cond),
            path,
            input_names=["mel", "timestep", "cond"],
            output_names=["noise_pred"],
            dynamic_axes={
                "mel": {0: "batch", 3: "time"},
                "timestep": {0: "batch"},
                "cond": {0: "batch"},
                "noise_pred": {0: "batch", 3: "time"},
            },
            opset_version=14,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = FaradayDiffusion(
        text_dim=512,
        speaker_dim=256,
        cond_dim=128,
        base_channels=64,
        num_steps=1000,
    ).to(device)

    total = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Target: ~20M")

    B = 1
    T = 256
    mel = torch.randn(B, 1, 80, T, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    text_emb = torch.randn(B, 512, device=device)
    speaker_emb = torch.randn(B, 256, device=device)

    # Forward pass
    with torch.no_grad():
        out = model(mel, t, text_emb, speaker_emb)
    assert out.shape == (B, 1, 80, T), f"Expected {(B, 1, 80, T)}, got {out.shape}"
    print(f"Forward pass OK: {out.shape}")

    # Enhance pass (10-step DDIM)
    with torch.no_grad():
        enhanced = model.enhance(mel, text_emb, speaker_emb, steps=10)
    assert enhanced.shape == (B, 1, 80, T), f"Expected {(B, 1, 80, T)}, got {enhanced.shape}"
    print(f"Enhance pass OK: {enhanced.shape}")

    # Training loss
    loss = model.training_loss(mel, text_emb, speaker_emb)
    print(f"Training loss OK: {loss.item():.4f}")

    # Memory footprint (fp16)
    param_bytes = total * 2  # fp16
    print(f"Estimated fp16 memory: {param_bytes / 1e6:.2f} MB")
