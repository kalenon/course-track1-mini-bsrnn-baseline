#!/usr/bin/env python3
"""Build reproducible TIMIT/WSJ + WHAM + RIR manifests and a data statement."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timit-root", type=Path, required=True)
    parser.add_argument("--wsj-root", type=Path, required=True)
    parser.add_argument("--wham-root", type=Path, required=True)
    parser.add_argument("--rir-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--validation-size", type=int, default=256)
    args = parser.parse_args()

    timit_train = audio_files(args.timit_root / "train")
    wsj_base = args.wsj_root / "wsj0_wav" / "wsj0"
    wsj_train = audio_files(wsj_base / "si_tr_s")
    wsj_valid = select_evenly(
        audio_files(wsj_base / "si_dt_05"), args.validation_size
    )
    noise_train = audio_files(args.wham_root / "wham_noise" / "tr")
    noise_valid = audio_files(args.wham_root / "wham_noise" / "cv")
    rir_paths = audio_files(args.rir_root)
    rir_valid = rir_paths[::10]
    valid_set = set(rir_valid)
    rir_train = [path for path in rir_paths if path not in valid_set]

    manifests = {
        "train_clean": records("timit", timit_train, args.timit_root)
        + records("wsj_train", wsj_train, args.wsj_root),
        "valid_clean": records("wsj_valid", wsj_valid, args.wsj_root),
        "train_noise": records("wham_train", noise_train, args.wham_root),
        "valid_noise": records("wham_valid", noise_valid, args.wham_root),
        "train_rir": records("rir_train", rir_train, args.rir_root),
        "valid_rir": records("rir_valid", rir_valid, args.rir_root),
    }
    for name, entries in manifests.items():
        write_scp(args.output_dir / f"{name}.scp", entries)

    statement = {
        "speech": {
            "train_sources": ["TIMIT train", "WSJ0 si_tr_s"],
            "validation_source": "WSJ0 si_dt_05 (even deterministic subset)",
        },
        "noise": {"train": "WHAM! tr", "validation": "WHAM! cv"},
        "rir": "record_RIR_with_T60_distance, stable 90/10 file split",
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
