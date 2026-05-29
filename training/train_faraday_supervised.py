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
from topology.barcode_loss import TopologicalLoss


class MelPairDataset(Dataset):
    def __init__(self, data_dir: str, preload: bool = True):
        self.paths = sorted(Path(data_dir).glob("*.pt"))
        if len(self.paths) == 0:
            raise ValueError(f"No .pt files found in {data_dir}")
        self.samples = []
        if preload:
            print(f"[Dataset] Pre-loading {len(self.paths)} samples into RAM...")
            for p in self.paths:
                data = torch.load(p, weights_only=True)
                self.samples.append({
                    "student_mel": data["student_mel"].squeeze(0),
                    "gt_mel": data["gt_mel"].squeeze(0),
                    "text_emb": data["text_emb"],
                    "speaker_emb": data["speaker_emb"],
                })
            print(f"[Dataset] Pre-loaded {len(self.samples)} samples")
        self.preload = preload

    def __len__(self):
        return len(self.paths) if not self.preload else len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            return self.samples[idx]
        data = torch.load(self.paths[idx], weights_only=True)
        return {
            "student_mel": data["student_mel"].squeeze(0),
            "gt_mel": data["gt_mel"].squeeze(0),
            "text_emb": data["text_emb"],
            "speaker_emb": data["speaker_emb"],
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


def train_epoch(model, loader, optimizer, device, grad_accum=1, epoch=0, output_dir=None, scaler=None, topo_loss_fn=None, topo_interval=10):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    step_losses = []
    use_amp = scaler is not None
    for step, batch in enumerate(tqdm(loader, desc="Train")):
        student_mel = batch["student_mel"].to(device)
        gt_mel = batch["gt_mel"].to(device)
        text_emb = batch["text_emb"].to(device)
        speaker_emb = batch["speaker_emb"].to(device)

        try:
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                # Topology loss is expensive (Ripser on CPU). Compute periodically.
                use_topo = (topo_loss_fn is not None) and (topo_interval > 0) and (step % topo_interval == 0)
                if use_topo:
                    pred_mel = model.supervised_enhance(student_mel, text_emb, speaker_emb)
                    loss = topo_loss_fn(pred_mel, gt_mel)
                else:
                    loss = model.supervised_training_loss(student_mel, gt_mel, text_emb, speaker_emb)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"\n[OOM] Step {step}, skipping batch. Error: {e}")
                continue
            raise

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
        step_losses.append(loss.item() * grad_accum)

        # Save checkpoint every 50 steps during epoch (crash protection)
        if output_dir and step > 0 and step % 50 == 0:
            ckpt_path = Path(output_dir) / f"epoch{epoch}_step{step}_emergency.pt"
            torch.save({
                "epoch": epoch,
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss.item(),
            }, ckpt_path)

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, scaler=None):
    model.eval()
    total_loss = 0.0
    use_amp = scaler is not None
    for batch in loader:
        student_mel = batch["student_mel"].to(device)
        gt_mel = batch["gt_mel"].to(device)
        text_emb = batch["text_emb"].to(device)
        speaker_emb = batch["speaker_emb"].to(device)

        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            loss = model.supervised_training_loss(student_mel, gt_mel, text_emb, speaker_emb)
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/faraday_pairs")
    parser.add_argument("--output_dir", default="./checkpoints/faraday")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (destiny threshold: 100)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0=main thread, faster on Windows)")
    parser.add_argument("--topo_interval", type=int, default=10, help="Compute topology loss every N steps (0=never)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[FaradayTrain] Device: {device}")

    # Speed settings: TF32 for faster matmul on Ada/Ampere
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    model = FaradayDiffusion(
        text_dim=512,
        speaker_dim=512,  # Match training data (SpeechT5 speaker embeddings are 512-dim)
        cond_dim=512,
        base_channels=192,  # ~509M params — true 400M+ class
    ).to(device)
    # Enable gradient checkpointing on U-Net for 8GB VRAM survival
    model.unet.use_checkpoint = True
    # Force checkpoint on all ResBlocks for maximum memory savings
    for name, module in model.unet.named_modules():
        if hasattr(module, 'checkpoint'):
            module.checkpoint = True

    # DISABLED: torch.compile causes TDR/thermal shutdown on RTX 4060 8GB
    # TF32 matmul precision is sufficient speedup (~1.3x) without instability
    print("[FaradayTrain] torch.compile DISABLED for stability. Using TF32 matmul.")

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[FaradayTrain] Params: {total:,} (~{total/1e6:.0f}M)")

    ds = MelPairDataset(args.data_dir, preload=True)
    train_size = int(0.9 * len(ds))
    val_size = len(ds) - train_size
    train_set, val_set = torch.utils.data.random_split(
        ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    num_workers = args.num_workers
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
    )

    # Use 8-bit AdamW to fit 400M params in 8GB VRAM (saves ~2.4GB optimizer state)
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
        print("[Optimizer] AdamW8bit (8-bit optimizer state)")
    except ImportError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
        print("[Optimizer] Standard AdamW (WARNING: may OOM on 8GB)")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Mixed precision scaler — cuts activation memory ~50%
    scaler = torch.amp.GradScaler(device='cuda') if device.type == "cuda" else None

    # Topology-aware loss (Betti number matching from persistent homology)
    topo_loss_fn = TopologicalLoss(betti_weight=0.1).to(device)

    best_val = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import time
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        train_loss = train_epoch(model, train_loader, optimizer, device, grad_accum=args.grad_accum, epoch=epoch, output_dir=args.output_dir, scaler=scaler, topo_loss_fn=topo_loss_fn, topo_interval=args.topo_interval)
        val_loss = validate(model, val_loader, device, scaler=scaler)
        scheduler.step()

        print(f"Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = output_dir / "best.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best checkpoint: {ckpt_path}")

        # Always save latest
        torch.save(model.state_dict(), output_dir / "last.pt")

        # Thermal cooldown between epochs to prevent TDR
        if device.type == "cuda":
            torch.cuda.synchronize()
            time.sleep(5)
            print("[Thermal] 5s cooldown complete")

    print("\n[FaradayTrain] Done.")


if __name__ == "__main__":
    main()
