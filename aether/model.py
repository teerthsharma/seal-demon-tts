"""AetherFilter wrapper: FilterNet + losses + export."""

import torch
import torch.nn as nn

from aether.filter_net import FilterNet
from aether.loss import TotalLoss
from aether.lattice_filter import count_parameters


class AetherFilter(nn.Module):
    def __init__(self, lr: float = 1e-4):
        super().__init__()
        self.filter_net = FilterNet()
        self.criterion = TotalLoss()
        self.lr = lr

    def forward(
        self,
        waveform: torch.Tensor,
        mel: torch.Tensor,
        speaker_emb: torch.Tensor,
        f0: torch.Tensor,
        energy: torch.Tensor,
        target_waveform: torch.Tensor = None,
    ):
        out = self.filter_net(waveform, mel, speaker_emb, f0, energy)
        if target_waveform is not None:
            loss = self.criterion(out, target_waveform)
            return out, loss
        return out

    def export_onnx(self, path: str):
        dummy_wav = torch.randn(1, 1, 24000)
        dummy_mel = torch.randn(1, 80, 100)
        dummy_spk = torch.randn(1, 192)
        dummy_f0 = torch.randn(1, 1, 100)
        dummy_energy = torch.randn(1, 1, 100)
        torch.onnx.export(
            self.filter_net,
            (dummy_wav, dummy_mel, dummy_spk, dummy_f0, dummy_energy),
            path,
            input_names=["waveform", "mel", "speaker_emb", "f0", "energy"],
            output_names=["waveform"],
            dynamic_axes={
                "waveform": {0: "batch", 2: "time"},
                "mel": {0: "batch", 2: "time"},
                "speaker_emb": {0: "batch"},
                "f0": {0: "batch", 2: "time"},
                "energy": {0: "batch", 2: "time"},
                "waveform_out": {0: "batch", 1: "time"},
            },
            opset_version=17,
        )


if __name__ == "__main__":
    model = AetherFilter()
    print(f"[AetherFilter] Params: {count_parameters(model.filter_net):,}")
    wav = torch.randn(1, 1, 24000)
    mel = torch.randn(1, 80, 100)
    spk = torch.randn(1, 192)
    f0 = torch.randn(1, 1, 100)
    energy = torch.randn(1, 1, 100)
    out, loss = model(wav, mel, spk, f0, energy, target_waveform=wav)
    print(f"Output shape: {out.shape}, loss: {loss.item():.4f}")
