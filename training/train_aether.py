#!/usr/bin/env python3
"""Train Aether differentiable filter post-net."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
import lightning as pl
from lightning.pytorch import Trainer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aether.model import AetherFilter
from training.common import count_parameters, setup_logging, get_dataloader, get_checkpoint_callbacks


class WaveformPairDataset(Dataset):
    """Dataset of (input_waveform, target_waveform, mel, speaker_emb, f0, energy)."""

    def __init__(self, data_dir: str, max_samples: int = 30_000):
        self.paths = list(Path(data_dir).glob("*.pt"))[:max_samples]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=True)
        return {
            "input_waveform": data["input_waveform"],   # [1, T]
            "target_waveform": data["target_waveform"], # [1, T]
            "mel": data["mel"],                         # [1, 80, T_mel]
            "speaker_emb": data.get("speaker_emb", torch.zeros(192)),
            "f0": data.get("f0", torch.zeros(1, data["mel"].size(-1))),
            "energy": data.get("energy", torch.zeros(1, data["mel"].size(-1))),
        }


def collate_fn(batch):
    max_t = max(b["input_waveform"].size(-1) for b in batch)
    max_mel_t = max(b["mel"].size(-1) for b in batch)
    pad_w = lambda t: torch.nn.functional.pad(t, (0, max_t - t.size(-1)))
    pad_m = lambda t: torch.nn.functional.pad(t, (0, max_mel_t - t.size(-1)))
    return {
        "input_waveform": torch.stack([pad_w(b["input_waveform"]) for b in batch]),
        "target_waveform": torch.stack([pad_w(b["target_waveform"]) for b in batch]),
        "mel": torch.stack([pad_m(b["mel"]) for b in batch]),
        "speaker_emb": torch.stack([b["speaker_emb"] for b in batch]),
        "f0": torch.stack([pad_m(b["f0"]) for b in batch]),
        "energy": torch.stack([pad_m(b["energy"]) for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints/aether")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=15_000)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    pl.seed_everything(42)

    model = AetherFilter(lr=args.lr)
    print(f"[Aether] Params: {count_parameters(model.filter_net):,}")

    ds = WaveformPairDataset(args.data_dir)
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_set, val_set = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = get_dataloader(train_set, args.batch_size, args.num_workers, shuffle=True)
    val_loader = get_dataloader(val_set, args.batch_size, args.num_workers, shuffle=False)

    logger = setup_logging("aether", save_dir=str(Path(args.output_dir).parent / "runs"))
    callbacks = get_checkpoint_callbacks(args.output_dir, every_n_hours=1)

    trainer = Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)

    export_dir = Path(args.output_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    model.export_onnx(str(export_dir / "aether.onnx"))
    print(f"[Export] Saved to {export_dir}")


if __name__ == "__main__":
    main()
