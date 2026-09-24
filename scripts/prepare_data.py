#!/usr/bin/env python3
"""Build LibriTTS + WHAM + DNS RIR manifests for the 16-kHz baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def audio_files(root: Path) -> list[Path]:
    paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in {".wav", ".flac"})
    if not paths:
        raise FileNotFoundError(f"No WAV/FLAC files found under {root}")
    return paths


def stable_uid(prefix: str, path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix()
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{path.stem.lower()}_{digest}"


def records(prefix: str, paths: list[Path], root: Path):
    return [(stable_uid(prefix, path, root), path.resolve()) for path in paths]


def write_scp(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for uid, audio_path in entries:
            handle.write(f"{uid} {audio_path}\n")


def describe(entries):
    seconds = 0.0
    sample_rates = set()
    channels = set()
    for _, path in entries:
        info = sf.info(path)
        seconds += info.frames / info.samplerate
        sample_rates.add(info.samplerate)
        channels.add(info.channels)
    return {
        "utterances": len(entries),
        "hours": round(seconds / 3600.0, 4),
        "sample_rates": sorted(sample_rates),
        "channels": sorted(channels),
    }


def select_evenly(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or count >= len(paths):
        return paths
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[index] for index in indices]


def libritts_splits(root: Path, prefix: str) -> dict[str, Path]:
    """Find standard splits, including under an extra LibriTTS/ directory."""
    splits = {}
    for directory in sorted(root.rglob("*")):
        if directory.is_dir() and directory.name.startswith(prefix):
            if directory.name in splits:
                raise ValueError(f"Duplicate LibriTTS split {directory.name}: {directory}")
            splits[directory.name] = directory
    return splits


def wham_split(root: Path, split: str) -> Path:
    candidates = [root / "wham_noise" / split, root / split]
    candidates.extend(path / split for path in root.rglob("wham_noise") if path.is_dir())
    matches = sorted({path.resolve() for path in candidates if path.is_dir()})
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one WHAM! {split} directory under {root}; found {matches}"
        )
    return matches[0]


def split_rirs(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Stable file-disjoint 90/10 split."""
    validation = paths[::10]
    validation_set = set(validation)
    training = [path for path in paths if path not in validation_set]
    if not training or not validation:
        raise ValueError("At least two RIR files are required for train/validation")
    return training, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libritts-root", type=Path, required=True)
    parser.add_argument("--wham-root", type=Path, required=True)
    parser.add_argument("--slr26-root", type=Path, required=True)
    parser.add_argument("--slr28-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--validation-size", type=int, default=256)
    args = parser.parse_args()

    train_splits = libritts_splits(args.libritts_root, "train-")
    valid_splits = libritts_splits(args.libritts_root, "dev-")
    if not train_splits or not valid_splits:
        raise FileNotFoundError(
            f"LibriTTS needs train-* and dev-* splits under {args.libritts_root}; "
            f"found train={list(train_splits)}, dev={list(valid_splits)}"
        )
    train_clean = sorted(path for split in train_splits.values() for path in audio_files(split))
    valid_clean = select_evenly(
        sorted(path for split in valid_splits.values() for path in audio_files(split)),
        args.validation_size,
    )
    noise_train = audio_files(wham_split(args.wham_root, "tr"))
    noise_valid = audio_files(wham_split(args.wham_root, "cv"))
    slr26_paths = set(audio_files(args.slr26_root))
    slr28_paths = set(audio_files(args.slr28_root))
    rir_train, rir_valid = split_rirs(sorted(slr26_paths | slr28_paths))

    manifests = {
        "train_clean": records("libritts_train", train_clean, args.libritts_root),
        "valid_clean": records("libritts_dev", valid_clean, args.libritts_root),
        "train_noise": records("wham_train", noise_train, args.wham_root),
        "valid_noise": records("wham_valid", noise_valid, args.wham_root),
        "train_rir": records("slr26", [p for p in rir_train if p in slr26_paths], args.slr26_root)
        + records("slr28", [p for p in rir_train if p in slr28_paths], args.slr28_root),
        "valid_rir": records("slr26", [p for p in rir_valid if p in slr26_paths], args.slr26_root)
        + records("slr28", [p for p in rir_valid if p in slr28_paths], args.slr28_root),
    }
    for name, entries in manifests.items():
        write_scp(args.output_dir / f"{name}.scp", entries)

    statement = {
        "speech": {
            "train_sources": sorted(train_splits),
            "validation_source": sorted(valid_splits),
        },
        "noise": {"train": "WHAM! tr", "validation": "WHAM! cv"},
        "rir": "DNS_ICASSP2021 SLR26 + SLR28, stable 90/10 file split",
        "redistribution": "No corpus audio is copied or redistributed.",
        "manifests": {name: describe(entries) for name, entries in manifests.items()},
        "mixing": {
            "target_sample_rate": 16000,
            "snr_db": [-5.0, 20.0],
            "reverberation_probability": 0.5,
            "segment_samples": 32000,
            "augmentations": ["bandwidth limitation", "clipping", "packet loss"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "data_statement.json").open("w", encoding="utf-8") as handle:
        json.dump(statement, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(statement["manifests"], ensure_ascii=False, indent=2))
    print(f"Wrote manifests to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
