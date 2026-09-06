#!/usr/bin/env python3
"""Train the fixed-16-kHz Mini-BSRNN baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_bsrnn.data import DynamicMixDataset  # noqa: E402
from mini_bsrnn.lightning import MiniBSRNNModule  # noqa: E402


def dataset_from_config(config, split, seed):
    data = config["data"]
    common = dict(
        segment_samples=config["segment_samples"],
        seed=seed,
        reverberation_probability=data["reverberation_probability"],
        snr_db=data["snr_db"],
        bandwidth_probability=data["bandwidth_probability"],
        clipping_probability=data["clipping_probability"],
        packet_loss_probability=data["packet_loss_probability"],
    )
    return DynamicMixDataset(
        clean_scp=data[f"{split}_clean_scp"],
        noise_scp=data[f"{split}_noise_scp"],
        rir_scp=data[f"{split}_rir_scp"],
        deterministic=split == "valid",
        **common,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mini_bsrnn.yaml")
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["sample_rate"] != 16_000:
        raise ValueError("The course baseline is fixed to 16 kHz")

    training = config["training"]
    seed = int(config["seed"])
    pl.seed_everything(seed, workers=True)
    train_set = dataset_from_config(config, "train", seed)
    valid_set = dataset_from_config(config, "valid", seed + 1_000_000)
    loader_args = dict(
        batch_size=training["batch_size_per_gpu"],
        num_workers=training["num_workers"],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=training["num_workers"] > 0,
    )
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    valid_loader = DataLoader(valid_set, shuffle=False, drop_last=False, **loader_args)

    model = MiniBSRNNModule(
        **config["model"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    output_dir = Path(training["output_dir"])
    logger = TensorBoardLogger(output_dir, name="tensorboard")
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    devices = args.devices if args.devices is not None else training["devices"]
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=devices if torch.cuda.is_available() else 1,
        strategy="ddp" if torch.cuda.is_available() and devices > 1 else "auto",
        max_epochs=training["epochs"],
        gradient_clip_val=training["gradient_clip"],
        precision=training["precision"],
        logger=logger,
        callbacks=[checkpoint, LearningRateMonitor(logging_interval="epoch")],
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=10,
    )
    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()

