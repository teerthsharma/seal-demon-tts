#!/usr/bin/env python3
"""Master training orchestrator.

Runs the full pipeline:
  1. Generate synthetic training data (if missing)
  2. Train Faraday (supervised mel enhancement)
  3. Train Aether (waveform post-filter)
  4. Copy best checkpoints to ./models/

Usage:
  python train_all.py --num_pairs 1000 --faraday_epochs 50 --aether_epochs 50
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list, desc: str):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"FAILED: {desc}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--faraday_epochs", type=int, default=50)
    parser.add_argument("--aether_epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data_dir = Path("./data")
    faraday_pairs = data_dir / "faraday_pairs"
    aether_pairs = data_dir / "aether_pairs"

    # Phase 1: Data generation
    if not faraday_pairs.exists() or len(list(faraday_pairs.glob("*.pt"))) < args.num_pairs:
        run_cmd(
            [
                sys.executable,
                "generate_training_data.py",
                "--num_pairs", str(args.num_pairs),
                "--output_dir", str(data_dir),
                "--device", args.device,
            ],
            "Phase 1: Generating synthetic training data",
        )
    else:
        print(f"[TrainAll] Found existing data: {len(list(faraday_pairs.glob('*.pt')))} pairs")

    # Phase 2: Train Faraday
    run_cmd(
        [
            sys.executable,
            "training/train_faraday_supervised.py",
            "--data_dir", str(faraday_pairs),
            "--output_dir", "./checkpoints/faraday",
            "--epochs", str(args.faraday_epochs),
            "--device", args.device,
        ],
        "Phase 2: Training Faraday (supervised mel enhancement)",
    )

    # Phase 3: Train Aether
    run_cmd(
        [
            sys.executable,
            "training/train_aether_supervised.py",
            "--data_dir", str(aether_pairs),
            "--output_dir", "./checkpoints/aether",
            "--epochs", str(args.aether_epochs),
            "--device", args.device,
        ],
        "Phase 3: Training Aether (waveform post-filter)",
    )

    # Phase 4: Copy checkpoints to models/
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)

    faraday_best = Path("./checkpoints/faraday/best.pt")
    aether_best = Path("./checkpoints/aether/best.pt")

    if faraday_best.exists():
        shutil.copy(faraday_best, models_dir / "faraday.pt")
        print(f"[TrainAll] Copied Faraday checkpoint to {models_dir / 'faraday.pt'}")

    if aether_best.exists():
        shutil.copy(aether_best, models_dir / "aether.pt")
        print(f"[TrainAll] Copied Aether checkpoint to {models_dir / 'aether.pt'}")

    print("\n" + "="*60)
    print("  ALL DONE")
    print("="*60)
    print("Run: python gui.py")


if __name__ == "__main__":
    main()
