#!/usr/bin/env python3
"""Batch offline inference with strict 16-kHz mono WAV input and output."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_bsrnn.model import MiniBSRNN_SE  # noqa: E402


def load_model(checkpoint_path: Path, device: torch.device) -> MiniBSRNN_SE:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    clean_state = {}
    for key, value in state.items():
        for prefix in ("module.", "model.", "se_model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith("bsrnn."):
            clean_state[key] = value
    model = MiniBSRNN_SE(embedding_dim=64, num_layers=2)
    model.load_state_dict(clean_state, strict=True)
    return model.to(device).eval()


def read_input_scp(path: Path):
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            uid, audio_path = line.strip().split(maxsplit=1)
            relative = Path(uid)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe ID at {path}:{line_number}: {uid}")
            if relative.suffix.lower() != ".wav":
                relative = relative.with_suffix(".wav")
            entries.append((uid, Path(audio_path), relative))
    return entries


def scan_input_dir(path: Path):
    entries = []
    for audio_path in sorted(path.rglob("*.wav")):
        relative = audio_path.relative_to(path)
        entries.append((relative.as_posix(), audio_path, relative))
    return entries


def load_strict_wav(path: Path) -> np.ndarray:
    info = sf.info(path)
    if info.format != "WAV" or info.samplerate != 16_000 or info.channels != 1:
        raise ValueError(
            f"Expected mono 16-kHz WAV, got format={info.format}, "
            f"sample_rate={info.samplerate}, channels={info.channels}: {path}"
        )
    audio, _ = sf.read(path, dtype="float32")
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid waveform: {path}")
    return audio


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-dir", type=Path)
    source.add_argument("--input-scp", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/mini_bsrnn_best.ckpt")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--peak", type=float, default=0.0,
        help="Optional output peak normalization; 0 disables it.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")
    entries = (
        scan_input_dir(args.input_dir)
        if args.input_dir is not None
        else read_input_scp(args.input_scp)
    )
    if not entries:
        raise ValueError("No input WAV files found")

    model = load_model(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "enhanced.scp"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for batch in tqdm(list(batches(entries, args.batch_size)), desc="Enhancing"):
            waveforms = [torch.from_numpy(load_strict_wav(path)) for _, path, _ in batch]
            lengths = torch.tensor([len(waveform) for waveform in waveforms], device=device)
            padded = pad_sequence(waveforms, batch_first=True).to(device)
            with torch.inference_mode():
                enhanced, _ = model(padded, lengths, 16_000)
            for index, (uid, _, relative) in enumerate(batch):
                output = enhanced[index, : lengths[index]].float().cpu().numpy()
                if args.peak > 0:
                    peak = float(np.max(np.abs(output)))
                    if peak > 0:
                        output = output * (args.peak / peak)
                output_path = args.output_dir / relative
                output_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(output_path, output, 16_000, format="WAV", subtype="PCM_16")
                if sf.info(output_path).frames != int(lengths[index]):
                    raise RuntimeError(f"Output length mismatch for {uid}")
                manifest.write(f"{uid} {output_path.resolve()}\n")
    print(f"Enhanced {len(entries)} files; manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
