#!/usr/bin/env python3
"""Train Faraday in supervised (deterministic) enhancement mode.

Uses plain PyTorch — no Lightning dependency.
Expects data in ./data/faraday_pairs/*.pt with keys:
  student_mel: [1, 80, T]
  gt_mel: [1, 80, T]
  text_emb: [512]
  speaker_emb: [512]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from faraday.model import FaradayDiffusion


class MelPairDataset(Dataset):
    def __init__(self, data_dir: str):
        self.paths = sorted(Path(data_dir).glob("*.pt"))
        if len(self.paths) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx])
        return {
            "student_mel": data["student_mel"].squeeze(0),  # [80, T]
            "gt_mel": data["gt_mel"].squeeze(0),  # [80, T]
            "text_emb": data["text_emb"],  # [512]
            "speaker_emb": data["speaker_emb"],  # [512]
        }


def collate_fn(batch):
    max_t = max(b["student_mel"].size(-1) for b in batch)
    pad = lambda t: F.pad(t, (0, max_t - t.size(-1)))
    return {
        "student_mel": torch.stack([pad(b["student_mel"]) for b in batch]).unsqueeze(1),  # [B, 1, 80, T]
        "gt_mel": torch.stack([pad(b["gt_mel"]) for b in batch]).unsqueeze(1),  # [B, 1, 80, T]
        "text_emb": torch.stack([b["text_emb"] for b in batch]),
        "speaker_emb": torch.stack([b["speaker_emb"] for b in batch]),
    }


def train_epoch(model, loader, optimizer, device, grad_accum=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc="Train")):
        student_mel = batch["student_mel"].to(device)
        gt_mel = batch["gt_mel"].to(device)
        text_emb = batch["text_emb"].to(device)
        speaker_emb = batch["speaker_emb"].to(device)

        loss = model.supervised_training_loss(student_mel, gt_mel, text_emb, speaker_emb)
        loss = loss / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        student_mel = batch["student_mel"].to(device)
        gt_mel = batch["gt_mel"].to(device)
        text_emb = batch["text_emb"].to(device)
        speaker_emb = batch["speaker_emb"].to(device)

        loss = model.supervised_training_loss(student_mel, gt_mel, text_emb, speaker_emb)
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/faraday_pairs")
    parser.add_argument("--output_dir", default="./checkpoints/faraday")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[FaradayTrain] Device: {device}")

    model = FaradayDiffusion(
        text_dim=512,
        speaker_dim=512,
        cond_dim=128,
        base_channels=64,
    ).to(device)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FaradayTrain] Params: {total:,}")

    ds = MelPairDataset(args.data_dir)
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_set, val_set = torch.utils.data.random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_accum=args.grad_accum)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = output_dir / "best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best checkpoint: {ckpt_path}")

        # Always save latest
        torch.save(model.state_dict(), output_dir / "last.pt")

    print("\n[FaradayTrain] Done.")


if __name__ == "__main__":
    main()
