#!/usr/bin/env python3
"""Train Faraday 2D mel diffusion enhancer."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
import lightning as pl
from lightning.pytorch import Trainer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from faraday.model import FaradayDiffusion
from training.common import count_parameters, setup_logging, get_dataloader, get_checkpoint_callbacks


class MelPairDataset(Dataset):
    """Dataset of (student_mel, gt_mel) pairs. Replace with real data loader."""

    def __init__(self, data_dir: str, max_samples: int = 50_000):
        self.paths = list(Path(data_dir).glob("*.pt"))[:max_samples]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx], weights_only=True)
        return {
            "student_mel": data["student_mel"],  # [1, 80, T]
            "gt_mel": data["gt_mel"],            # [1, 80, T]
            "text_emb": data.get("text_emb", torch.zeros(512)),
            "speaker_emb": data.get("speaker_emb", torch.zeros(192)),
        }


def collate_fn(batch):
    max_t = max(b["student_mel"].size(-1) for b in batch)
    pad = lambda t: torch.nn.functional.pad(t, (0, max_t - t.size(-1)))
    return {
        "student_mel": torch.stack([pad(b["student_mel"]) for b in batch]),
        "gt_mel": torch.stack([pad(b["gt_mel"]) for b in batch]),
        "text_emb": torch.stack([b["text_emb"] for b in batch]),
        "speaker_emb": torch.stack([b["speaker_emb"] for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="checkpoints/faraday")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=30_000)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    pl.seed_everything(42)

    model = FaradayDiffusion(cond_dim=512 + 192, lr=args.lr)
    print(f"[Faraday] Params: {count_parameters(model.unet):,}")

    ds = MelPairDataset(args.data_dir)
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_set, val_set = torch.utils.data.random_split(ds, [train_size, val_size])

    train_loader = get_dataloader(train_set, args.batch_size, args.num_workers, shuffle=True)
    val_loader = get_dataloader(val_set, args.batch_size, args.num_workers, shuffle=False)

    logger = setup_logging("faraday", save_dir=str(Path(args.output_dir).parent / "runs"))
    callbacks = get_checkpoint_callbacks(args.output_dir, every_n_hours=1)

    trainer = Trainer(
        max_steps=args.max_steps,
        accelerator="gpu",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=logger,
        val_check_interval=2000,
        check_val_every_n_epoch=None,
    )

    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)

    export_dir = Path(args.output_dir) / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    model.export_onnx(str(export_dir / "faraday.onnx"))
    print(f"[Export] Saved to {export_dir}")


if __name__ == "__main__":
    main()
