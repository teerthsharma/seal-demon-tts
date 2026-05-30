"""Shared training utilities for demon-tts."""

import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def count_parameters(model: torch.nn.Module) -> int:
    """Return total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logging(name: str, save_dir: str = "runs") -> TensorBoardLogger:
    """Create a TensorBoard logger."""
    return TensorBoardLogger(save_dir=save_dir, name=name)


def get_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    collate_fn=None,
) -> DataLoader:
    """Build a standard DataLoader with best-practice defaults."""
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(**kwargs)


def get_checkpoint_callbacks(output_dir: str, every_n_hours: Optional[int] = 1):
    """Return standard checkpoint + early-stop callbacks.

    Includes both val_loss-monitored AND train_loss-monitored checkpoints
    so progress is saved even before the first validation run.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [
        # Best-by-validation checkpoint (the quality gate)
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch}-{step}",
            save_top_k=2,
            monitor="val_loss",
            mode="min",
            every_n_train_steps=1000,
            save_last=True,
        ),
        # Frequent train-loss checkpoint (the safety net — always saves)
        ModelCheckpoint(
            dirpath=output_dir,
            filename="train-{epoch}-{step}",
            save_top_k=-1,  # keep all
            monitor="train_loss",
            mode="min",
            every_n_train_steps=250,
        ),
    ]
    if every_n_hours:
        callbacks.append(
            ModelCheckpoint(
                dirpath=output_dir,
                filename="hourly-{epoch}-{step}",
                train_time_interval=timedelta(hours=every_n_hours),
                save_top_k=-1,  # keep all hourly
            )
        )
    callbacks.append(EarlyStopping(monitor="val_loss", patience=10, mode="min"))
    return callbacks
