#!/usr/bin/env python3
"""
Sequential Training Scheduler for Demon-TTS
============================================
Trains Faraday first, then Aether automatically.
Resumes from checkpoints if interrupted.
Single-GPU safe - no CUDA contention.

Usage:
    python train_scheduler.py --faraday-epochs 100 --aether-epochs 100
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_training(script_path: str, epochs: int, resume_ckpt: str = None, extra_args: list = None):
    """Run a training script and block until completion."""
    cmd = [sys.executable, script_path, "--epochs", str(epochs)]
    if resume_ckpt and Path(resume_ckpt).exists():
        cmd += ["--resume", resume_ckpt]
    if extra_args:
        cmd += extra_args
    
    print(f"\n{'='*60}")
    print(f"[Scheduler] Starting: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def find_latest_checkpoint(checkpoint_dir: str, pattern: str = "*.pt") -> str:
    """Find the most recent checkpoint in a directory."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(checkpoints[0]) if checkpoints else None


def find_latest_checkpoint_any(checkpoint_dir: Path, patterns: list[str]) -> str:
    """Find the newest checkpoint matching any of the provided patterns.

    Patterns are checked IN ORDER — e.g. last.pt is preferred over
    epoch*_emergency.pt even if the emergency file has a newer mtime.
    This prevents resuming from stale emergency checkpoints.
    """
    if not checkpoint_dir.exists():
        return None
    for pattern in patterns:
        matches = sorted(checkpoint_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matches:
            return str(matches[0])
    return None


def main():
    parser = argparse.ArgumentParser(description="Sequential Demon-TTS Training Scheduler")
    parser.add_argument("--faraday-epochs", type=int, default=100, help="Epochs for Faraday")
    parser.add_argument("--aether-epochs", type=int, default=100, help="Epochs for Aether")
    parser.add_argument("--skip-faraday", action="store_true", help="Skip Faraday, train Aether only")
    parser.add_argument("--skip-aether", action="store_true", help="Skip Aether, train Faraday only")
    parser.add_argument("--faraday-resume", type=str, default=None, help="Resume Faraday from checkpoint")
    parser.add_argument("--aether-resume", type=str, default=None, help="Resume Aether from checkpoint")
    parser.add_argument("--auto-resume", action="store_true", help="Auto-find latest checkpoint to resume")
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    faraday_script = base_dir / "training" / "train_faraday_supervised.py"
    aether_script = base_dir / "training" / "train_aether_supervised.py"
    
    # Auto-resume: find latest checkpoints
    faraday_ckpt = args.faraday_resume
    aether_ckpt = args.aether_resume
    if args.auto_resume:
        if not faraday_ckpt:
            faraday_ckpt = find_latest_checkpoint_any(
                base_dir / "checkpoints" / "faraday",
                ["last.pt", "epoch*_emergency.pt", "epoch*.pt", "best.pt"],
            )
        if not aether_ckpt:
            aether_ckpt = find_latest_checkpoint_any(
                base_dir / "checkpoints" / "aether",
                ["last.pt", "epoch*_emergency.pt", "epoch*.pt", "best.pt"],
            )
        if faraday_ckpt:
            print(f"[Scheduler] Auto-resume Faraday: {faraday_ckpt}")
        if aether_ckpt:
            print(f"[Scheduler] Auto-resume Aether: {aether_ckpt}")
    
    exit_code = 0
    
    # Phase 1: Faraday
    if not args.skip_faraday:
        exit_code = run_training(
            str(faraday_script),
            args.faraday_epochs,
            resume_ckpt=faraday_ckpt,
            extra_args=[
                "--batch_size", "1",
                "--grad_accum", "8",
                "--topo_interval", "0",
            ]
        )
        if exit_code != 0:
            print(f"[Scheduler] Faraday training failed with code {exit_code}")
            return exit_code
        print("[Scheduler] Faraday training complete!")
    
    # Phase 2: Aether
    if not args.skip_aether:
        # Thermal cooldown before switching models
        print("[Scheduler] GPU cooldown (10s) before Aether...")
        time.sleep(10)
        
        exit_code = run_training(
            str(aether_script),
            args.aether_epochs,
            resume_ckpt=aether_ckpt,
            extra_args=["--batch_size", "1", "--grad_accum", "4"]
        )
        if exit_code != 0:
            print(f"[Scheduler] Aether training failed with code {exit_code}")
            return exit_code
        print("[Scheduler] Aether training complete!")
    
    print("\n" + "="*60)
    print("[Scheduler] ALL TRAINING COMPLETE!")
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
