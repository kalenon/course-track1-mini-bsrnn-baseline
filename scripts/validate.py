#!/usr/bin/env python3
"""Enhance the prepared 1000-pair validation set with a Mini-BSRNN checkpoint."""

from __future__ import annotations

import argparse
import json
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


def read_scp(path: Path) -> list[tuple[str, Path]]:
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                uid, audio_path = line.strip().split(maxsplit=1)
            except ValueError as error:
                raise ValueError(f"Malformed line {line_number} in {path}") from error
            entries.append((uid, Path(audio_path)))
    if not entries:
        raise ValueError(f"No entries in {path}")
    return entries


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


def load_validation_wav(path: Path) -> np.ndarray:
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


def safe_filename(uid: str) -> str:
    path = Path(uid)
    if path.is_absolute() or len(path.parts) != 1 or uid in {"", ".", ".."}:
        raise ValueError(f"Validation ID must be a filename-safe token: {uid!r}")
    return f"{uid}.wav"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/validation_1000"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("checkpoints/mini_bsrnn_best.ckpt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/validation_1000")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Enhance only the first N pairs for an execution smoke test.",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")

    clean_entries = read_scp(args.data_dir / "clean.scp")
    noisy_entries = read_scp(args.data_dir / "noisy.scp")
    clean_by_id = dict(clean_entries)
    noisy_ids = [uid for uid, _ in noisy_entries]
    if set(clean_by_id) != set(noisy_ids) or len(noisy_ids) != len(set(noisy_ids)):
        raise RuntimeError("Clean and noisy validation IDs do not match")
    if args.limit is None and len(noisy_entries) != 1000:
        raise RuntimeError(
            f"Expected 1000 validation pairs; found {len(noisy_entries)}. "
            "Use --limit only for a smoke test."
        )
    if args.limit is not None:
        noisy_entries = noisy_entries[: args.limit]

    model = load_model(args.checkpoint, device)
    enhanced_dir = args.output_dir / "enhanced"
    manifest_dir = args.output_dir / "manifests"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    enhanced_lines = []
    clean_lines = []
    noisy_lines = []
    batch_count = (len(noisy_entries) + args.batch_size - 1) // args.batch_size
    for batch in tqdm(
        batches(noisy_entries, args.batch_size), total=batch_count,
        desc="Enhancing validation set",
    ):
        waveforms = [torch.from_numpy(load_validation_wav(path)) for _, path in batch]
        lengths = torch.tensor([len(waveform) for waveform in waveforms], device=device)
        padded = pad_sequence(waveforms, batch_first=True).to(device)
        with torch.inference_mode():
            enhanced, _ = model(padded, lengths, 16_000)
        for index, (uid, noisy_path) in enumerate(batch):
            length = int(lengths[index].item())
            clean = load_validation_wav(clean_by_id[uid])
            if len(clean) != length:
                raise RuntimeError(f"Clean/noisy length mismatch for {uid}")
            output_path = (enhanced_dir / safe_filename(uid)).resolve()
            output = enhanced[index, :length].float().cpu().numpy()
            sf.write(output_path, output, 16_000, format="WAV", subtype="PCM_16")
            output_info = sf.info(output_path)
            if output_info.frames != length or output_info.samplerate != 16_000:
                raise RuntimeError(f"Enhanced output format mismatch for {uid}")
            clean_lines.append(f"{uid} {clean_by_id[uid].resolve()}\n")
            noisy_lines.append(f"{uid} {noisy_path.resolve()}\n")
            enhanced_lines.append(f"{uid} {output_path}\n")

    (manifest_dir / "clean.scp").write_text("".join(clean_lines), encoding="utf-8")
    (manifest_dir / "noisy.scp").write_text("".join(noisy_lines), encoding="utf-8")
    (manifest_dir / "enhanced.scp").write_text(
        "".join(enhanced_lines), encoding="utf-8"
    )
    summary = {
        "num_files": len(enhanced_lines),
        "sample_rate": 16_000,
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_data": str(args.data_dir.resolve()),
    }
    (args.output_dir / "validation_run.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
