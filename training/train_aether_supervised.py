#!/usr/bin/env python3
"""Train Aether filter net in supervised mode.

Plain PyTorch — no Lightning dependency.
Expects data in ./data/aether_pairs/*.pt with keys:
  input_waveform: [1, T]
  target_waveform: [1, T]
  mel: [1, 80, T_mel]
  speaker_emb: [192]
  f0: [1, T_mel]
  energy: [1, T_mel]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from aether.model import AetherFilter


class WaveformPairDataset(Dataset):
    def __init__(self, data_dir: str):
        self.paths = sorted(Path(data_dir).glob("*.pt"))
        if len(self.paths) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = torch.load(self.paths[idx])
        return {
            "input_waveform": data["input_waveform"].squeeze(0),  # [T]
            "target_waveform": data["target_waveform"].squeeze(0),  # [T]
            "mel": data["mel"].squeeze(0),  # [80, T_mel]
            "speaker_emb": data["speaker_emb"],  # [192]
            "f0": data["f0"].squeeze(0),  # [T_mel]
            "energy": data["energy"].squeeze(0),  # [T_mel]
        }


def collate_fn(batch):
    max_w = max(b["input_waveform"].size(-1) for b in batch)
    max_m = max(b["mel"].size(-1) for b in batch)
    pad_w = lambda t: F.pad(t, (0, max_w - t.size(-1)))
    pad_m = lambda t: F.pad(t, (0, max_m - t.size(-1)))
    return {
        "input_waveform": torch.stack([pad_w(b["input_waveform"]) for b in batch]).unsqueeze(1),  # [B, 1, T]
        "target_waveform": torch.stack([pad_w(b["target_waveform"]) for b in batch]).unsqueeze(1),  # [B, 1, T]
        "mel": torch.stack([pad_m(b["mel"]) for b in batch]),
        "speaker_emb": torch.stack([b["speaker_emb"] for b in batch]),
        "f0": torch.stack([pad_m(b["f0"]) for b in batch]),  # [B, 1, T_mel]
        "energy": torch.stack([pad_m(b["energy"]) for b in batch]),  # [B, 1, T_mel]
    }


def train_epoch(model, loader, optimizer, device, grad_accum=1, scaler=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    use_amp = scaler is not None
    for step, batch in enumerate(tqdm(loader, desc="Train")):
        wav_in = batch["input_waveform"].to(device)
        wav_tgt = batch["target_waveform"].to(device)
        mel = batch["mel"].to(device)
        spk = batch["speaker_emb"].to(device)
        f0 = batch["f0"].to(device)
        energy = batch["energy"].to(device)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            _, loss = model(wav_in, mel, spk, f0, energy, target_waveform=wav_tgt)
        loss = loss / grad_accum
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, scaler=None):
    model.eval()
    total_loss = 0.0
    use_amp = scaler is not None
    for batch in loader:
        wav_in = batch["input_waveform"].to(device)
        wav_tgt = batch["target_waveform"].to(device)
        mel = batch["mel"].to(device)
        spk = batch["speaker_emb"].to(device)
        f0 = batch["f0"].to(device)
        energy = batch["energy"].to(device)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            _, loss = model(wav_in, mel, spk, f0, energy, target_waveform=wav_tgt)
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/aether_pairs")
    parser.add_argument("--output_dir", default="./checkpoints/aether")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[AetherTrain] Device: {device}")

    torch.set_float32_matmul_precision('high')

    model = AetherFilter(lr=args.lr).to(device)
    # DISABLED: torch.compile causes TDR/thermal shutdown on RTX 4060 8GB
    print("[AetherTrain] torch.compile DISABLED for stability. Using TF32 matmul.")

    total = sum(p.numel() for p in model.filter_net.parameters() if p.requires_grad)
    print(f"[AetherTrain] Params: {total:,}")

    ds = WaveformPairDataset(args.data_dir)
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_set, val_set = torch.utils.data.random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    num_workers = args.num_workers if args.num_workers > 0 else 28
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
        print("[Optimizer] AdamW8bit (8-bit optimizer state)")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
        print("[Optimizer] Standard AdamW")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler(device='cuda') if device.type == "cuda" else None

    best_val = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import time
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_accum=args.grad_accum, scaler=scaler)
        val_loss = validate(model, val_loader, device, scaler=scaler)
        scheduler.step()

        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = output_dir / "best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best checkpoint: {ckpt_path}")

        torch.save(model.state_dict(), output_dir / "last.pt")

        # Thermal cooldown between epochs to prevent TDR
        if device.type == "cuda":
            torch.cuda.synchronize()
            time.sleep(5)
            print("[Thermal] 5s cooldown complete")

    print("\n[AetherTrain] Done.")


if __name__ == "__main__":
    main()
