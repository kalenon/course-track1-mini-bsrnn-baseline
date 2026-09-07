#!/usr/bin/env python3
"""Validate the labeled 1000-pair archive and convert both sides to 16 kHz."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


EXPECTED_SHA256 = "edd77dccb6cc1d7c273f2a05a8daee0d26956bc748ac2472a1f9f7305a896080"
ARCHIVE_ROOT = "validation_leaderboard_with_label"
DATA_ROOT = f"{ARCHIVE_ROOT}/data/validation_leaderboard"
AUDIO_ROOT = f"{ARCHIVE_ROOT}/simulation_validation_leaderboard"
TARGET_SAMPLE_RATE = 16_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mapping(text: str, value_type=str) -> dict:
    result = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            key, value = line.split(maxsplit=1)
        except ValueError as error:
            raise ValueError(f"Malformed metadata line {line_number}") from error
        result[key] = value_type(value)
    return result


def audio_members(archive: ZipFile, kind: str) -> dict[str, str]:
    prefix = f"{AUDIO_ROOT}/{kind}/0/"
    result = {
        Path(name).stem: name
        for name in archive.namelist()
        if name.startswith(prefix) and name.lower().endswith(".flac")
    }
    if len(result) != 1000:
        raise RuntimeError(f"Expected 1000 {kind} files, found {len(result)}")
    return result


def read_archive_audio(archive: ZipFile, member: str) -> tuple[np.ndarray, int]:
    with archive.open(member) as handle:
        audio, sample_rate = sf.read(
            io.BytesIO(handle.read()), dtype="float32", always_2d=True
        )
    audio = audio.mean(axis=1)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite samples in {member}")
    return audio, int(sample_rate)


def resample_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)
    divisor = np.gcd(sample_rate, TARGET_SAMPLE_RATE)
    result = resample_poly(
        audio, TARGET_SAMPLE_RATE // divisor, sample_rate // divisor
    ).astype(np.float32)
    target_length = round(len(audio) * TARGET_SAMPLE_RATE / sample_rate)
    if len(result) < target_length:
        result = np.pad(result, (0, target_length - len(result)))
    return result[:target_length]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("data/downloads/validation_1000.zip")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/validation_1000")
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Prepare only the first N pairs for an execution smoke test.",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    archive_hash = sha256(args.archive)
    if archive_hash != EXPECTED_SHA256:
        raise RuntimeError(
            f"Archive SHA256 mismatch: expected {EXPECTED_SHA256}, got {archive_hash}"
        )

    with ZipFile(args.archive) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad_member}")
        noisy_members = audio_members(archive, "noisy")
        clean_members = audio_members(archive, "clean")
        original_fs = read_mapping(
            archive.read(f"{DATA_ROOT}/utt2fs").decode("utf-8"), int
        )
        expected_ids = set(read_mapping(
            archive.read(f"{DATA_ROOT}/wav.scp").decode("utf-8")
        ))
        if not (
            expected_ids == set(clean_members) == set(noisy_members) == set(original_fs)
        ):
            raise RuntimeError("Audio and metadata utterance IDs do not match")

        ordered_ids = sorted(
            expected_ids, key=lambda uid: int(uid.removeprefix("fileid_"))
        )
        source_pair_count = len(ordered_ids)
        if source_pair_count != 1000:
            raise RuntimeError(f"Expected 1000 pairs, found {source_pair_count}")
        if args.limit is not None:
            ordered_ids = ordered_ids[: args.limit]

        clean_dir = args.output_dir / "clean"
        noisy_dir = args.output_dir / "noisy"
        clean_dir.mkdir(parents=True, exist_ok=True)
        noisy_dir.mkdir(parents=True, exist_ok=True)
        clean_lines = []
        noisy_lines = []
        sample_rate_counts = Counter(original_fs.values())
        for uid in tqdm(ordered_ids, desc="Preparing validation pairs"):
            clean, clean_fs = read_archive_audio(archive, clean_members[uid])
            noisy, noisy_fs = read_archive_audio(archive, noisy_members[uid])
            if clean_fs != noisy_fs or clean_fs != original_fs[uid]:
                raise RuntimeError(f"Sample-rate mismatch for {uid}")
            if len(clean) != len(noisy):
                raise RuntimeError(f"Original length mismatch for {uid}")
            clean = resample_16k(clean, clean_fs)
            noisy = resample_16k(noisy, noisy_fs)
            if len(clean) != len(noisy):
                raise RuntimeError(f"Resampled length mismatch for {uid}")

            clean_path = (clean_dir / f"{uid}.wav").resolve()
            noisy_path = (noisy_dir / f"{uid}.wav").resolve()
            sf.write(clean_path, clean, TARGET_SAMPLE_RATE, subtype="FLOAT")
            sf.write(noisy_path, noisy, TARGET_SAMPLE_RATE, subtype="FLOAT")
            clean_lines.append(f"{uid} {clean_path}\n")
            noisy_lines.append(f"{uid} {noisy_path}\n")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "clean.scp").write_text("".join(clean_lines), encoding="utf-8")
    (args.output_dir / "noisy.scp").write_text("".join(noisy_lines), encoding="utf-8")
    summary = {
        "archive_sha256": archive_hash,
        "source_pair_count": source_pair_count,
        "prepared_pair_count": len(ordered_ids),
        "original_sample_rate_counts": dict(sorted(sample_rate_counts.items())),
        "evaluation_sample_rate": TARGET_SAMPLE_RATE,
        "conversion": "clean and noisy are independently resampled to 16 kHz",
        "audio_format": "mono 32-bit float WAV",
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
