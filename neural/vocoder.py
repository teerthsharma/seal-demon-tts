"""Lightweight HiFi-GAN generator (~14M params)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=dilation * (kernel_size // 2), dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        return x + residual


class UpsampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample_ratio: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(in_ch, out_ch, upsample_ratio * 2, stride=upsample_ratio, padding=upsample_ratio // 2)
        self.resblocks = nn.ModuleList([ResBlock(out_ch, dilation=3**i) for i in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv(x), 0.1)
        for rb in self.resblocks:
            x = rb(x)
        return x


class HiFiGenerator(nn.Module):
    """Small HiFi-GAN generator."""

    def __init__(self, in_ch: int = 80, upsample_rates: list = [8, 8, 2, 2], ngf: int = 256):
        super().__init__()
        self.pre = nn.Conv1d(in_ch, ngf, kernel_size=7, padding=3)
        layers = []
        ch = ngf
        for r in upsample_rates:
            layers.append(UpsampleBlock(ch, ch // 2, r))
            ch = ch // 2
        self.ups = nn.ModuleList(layers)
        self.post = nn.Conv1d(ch, 1, kernel_size=7, padding=3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, 80, T]
        x = F.leaky_relu(self.pre(mel), 0.1)
        for up in self.ups:
            x = up(x)
        x = torch.tanh(self.post(x))
        return x.squeeze(1)  # [B, T]

    def export_onnx(self, path: str):
        dummy = torch.randn(1, 80, 100)
        torch.onnx.export(
            self,
            dummy,
            path,
            input_names=["mel"],
            output_names=["waveform"],
            dynamic_axes={"mel": {0: "batch", 2: "time"}, "waveform": {0: "batch", 1: "time"}},
            opset_version=17,
        )


if __name__ == "__main__":
    model = HiFiGenerator()
    print(f"[Vocoder] Params: {count_parameters(model):,}")
    mel = torch.randn(1, 80, 100)
    wav = model(mel)
    print(f"Output shape: {wav.shape}")
