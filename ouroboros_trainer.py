"""Ouroboros Trainer: Self-improving training loop for DemonTTS.

Pass 1: Train on SpeechT5-generated synthetic data (existing pairs).
Pass 2: Use trained Faraday+Aether to generate BETTER synthetic data from book text.
Pass 3: Retrain on the enhanced synthetic data.

The snake eats its own tail and gets stronger.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, desc):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"FAILED: {desc}")
        return False
    return True


def ouroboros_pass(pass_num, num_pairs, faraday_epochs, aether_epochs):
    """One pass of the Ouroboros loop."""
    print(f"\n🐍 OUROBOROS PASS {pass_num} 🐍")

    if pass_num == 1:
        data_source = "./book_parsed/"
        print("  Using SpeechT5 teacher for synthetic data...")
    else:
        data_source = "./ouroboros_generated/"
        print("  Using trained DemonTTS for self-generated data...")

        gen_cmd = [
            sys.executable,
            "generate_ouroboros_data.py",
            "--input_dir", "./book_parsed/",
            "--output_dir", data_source,
            "--num_pairs", str(num_pairs),
            "--models_dir", "./models/",
        ]
        if not run_cmd(gen_cmd, f"Pass {pass_num}: Generating self-improved data"):
            return False

    pairs_cmd = [
        sys.executable,
        "generate_training_data.py",
        "--text_source", data_source,
        "--output_dir", "./data",
        "--num_pairs", str(num_pairs),
    ]
    if not run_cmd(pairs_cmd, f"Pass {pass_num}: Building training pairs"):
        return False

    faraday_cmd = [
        sys.executable,
        "training/train_faraday_supervised.py",
        "--data_dir", "./data/faraday_pairs",
        "--output_dir", f"./checkpoints/faraday_pass{pass_num}",
        "--batch_size", "1",
        "--grad_accum", "8",
        "--epochs", str(faraday_epochs),
    ]
    if not run_cmd(faraday_cmd, f"Pass {pass_num}: Training Faraday"):
        return False

    aether_cmd = [
        sys.executable,
        "training/train_aether_supervised.py",
        "--data_dir", "./data/aether_pairs",
        "--output_dir", f"./checkpoints/aether_pass{pass_num}",
        "--batch_size", "1",
        "--grad_accum", "4",
        "--epochs", str(aether_epochs),
    ]
    if not run_cmd(aether_cmd, f"Pass {pass_num}: Training Aether"):
        return False

    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)

    faraday_best = Path(f"./checkpoints/faraday_pass{pass_num}/best.pt")
    aether_best = Path(f"./checkpoints/aether_pass{pass_num}/best.pt")

    if faraday_best.exists():
        shutil.copy(faraday_best, models_dir / "faraday.pt")
    if aether_best.exists():
        shutil.copy(aether_best, models_dir / "aether.pt")

    print(f"\n✅ Pass {pass_num} complete. Checkpoints updated.")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--num_pairs", type=int, default=1000)
    parser.add_argument("--faraday_epochs", type=int, default=100)
    parser.add_argument("--aether_epochs", type=int, default=100)
    args = parser.parse_args()

    for p in range(1, args.passes + 1):
        if not ouroboros_pass(p, args.num_pairs, args.faraday_epochs, args.aether_epochs):
            print(f"Ouroboros failed at pass {p}")
            sys.exit(1)

    print("\n🐍 OUROBOROS COMPLETE 🐍")
    print("The snake has consumed itself and emerged stronger.")
    print("Run: python gui.py")


if __name__ == "__main__":
    main()
