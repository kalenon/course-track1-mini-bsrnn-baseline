#!/usr/bin/env python3
"""Run deterministic inference on the configured internal validation set."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import soundfile as sf
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_bsrnn.data import DynamicMixDataset  # noqa: E402
from mini_bsrnn.model import MiniBSRNN_SE  # noqa: E402


def load_model(checkpoint_path: Path, device: torch.device) -> MiniBSRNN_SE:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    model = MiniBSRNN_SE(embedding_dim=64, num_layers=2)
    expected = set(model.state_dict())
    clean_state = {}
    for original_key, value in state.items():
        key = original_key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "model.", "se_model."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    changed = True
                    break
        if key in expected:
            clean_state[key] = value
    missing = expected.difference(clean_state)
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Checkpoint is missing {len(missing)} model tensors: {preview}")
    model.load_state_dict(clean_state, strict=True)
    return model.to(device).eval()


def build_validation_dataset(config: dict) -> DynamicMixDataset:
    data = config["data"]
    return DynamicMixDataset(
        clean_scp=data["valid_clean_scp"],
        noise_scp=data["valid_noise_scp"],
        rir_scp=data["valid_rir_scp"],
        segment_samples=config["segment_samples"],
        seed=int(config["seed"]) + 1_000_000,
        deterministic=True,
        reverberation_probability=data["reverberation_probability"],
        snr_db=data["snr_db"],
        bandwidth_probability=data["bandwidth_probability"],
        clipping_probability=data["clipping_probability"],
        packet_loss_probability=data["packet_loss_probability"],
    )


def safe_filename(uid: str) -> str:
    path = Path(uid)
    if path.is_absolute() or len(path.parts) != 1 or uid in {"", ".", ".."}:
        raise ValueError(f"Validation ID must be a filename-safe token: {uid!r}")
    return f"{uid}.wav"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mini_bsrnn.yaml"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/mini_bsrnn_best.ckpt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("runs/validation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("--batch-size must be positive and --num-workers non-negative")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")

    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["sample_rate"] != 16_000:
        raise ValueError("The course baseline is fixed to 16 kHz")
    dataset = build_validation_dataset(config)
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = load_model(args.checkpoint, device)

    audio_dirs = {
        name: args.output_dir / name for name in ("clean", "noisy", "enhanced")
    }
    manifest_dir = args.output_dir / "manifests"
    for directory in (*audio_dirs.values(), manifest_dir):
        directory.mkdir(parents=True, exist_ok=True)

    count = 0
    with ExitStack() as stack:
        manifests = {
            name: stack.enter_context(
                (manifest_dir / f"{name}.scp").open("w", encoding="utf-8")
            )
            for name in audio_dirs
        }
        for batch in tqdm(loader, desc="Validating"):
            noisy = batch["noisy"].float().to(device)
            lengths = torch.as_tensor(batch["length"], device=device)
            with torch.inference_mode():
                enhanced, _ = model(noisy, lengths, 16_000)
            arrays = {
                "clean": batch["clean"].float().cpu(),
                "noisy": batch["noisy"].float().cpu(),
                "enhanced": enhanced.float().cpu(),
            }
            for index, uid in enumerate(batch["id"]):
                filename = safe_filename(uid)
                length = int(lengths[index].item())
                for name, values in arrays.items():
                    output_path = audio_dirs[name] / filename
                    sf.write(
                        output_path,
                        values[index, :length].numpy(),
                        16_000,
                        format="WAV",
                        subtype="PCM_16",
                    )
                    manifests[name].write(f"{uid} {output_path.resolve()}\n")
                count += 1

    summary = {
        "num_files": count,
        "sample_rate": 16_000,
        "segment_samples": int(config["segment_samples"]),
        "seed": int(config["seed"]) + 1_000_000,
        "checkpoint": str(args.checkpoint.resolve()),
    }
    with (args.output_dir / "validation_run.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
