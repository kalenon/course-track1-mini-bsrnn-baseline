#!/usr/bin/env python3
"""Evaluate PESQ, ESTOI, SI-SDR and UTMOS under the course protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm


PENALTY = {"PESQ": -0.5, "ESTOI": 0.0, "SI_SDR": -50.0, "UTMOS": 1.0}


def read_scp(path: Path):
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            uid, audio_path = line.strip().split(maxsplit=1)
            entries.append((uid, Path(audio_path)))
    if not entries:
        raise ValueError(f"No entries in {path}")
    return entries


def load_course_wav(path: Path):
    info = sf.info(path)
    if info.format != "WAV" or info.samplerate != 16_000 or info.channels != 1:
        raise ValueError(
            f"not a mono 16-kHz WAV (format={info.format}, "
            f"sample_rate={info.samplerate}, channels={info.channels})"
        )
    audio, _ = sf.read(path, dtype="float32")
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError("invalid samples")
    return audio


def si_sdr(reference, estimate, eps=1.0e-8):
    reference = reference.astype(np.float64) - np.mean(reference)
    estimate = estimate.astype(np.float64) - np.mean(estimate)
    projection = np.dot(estimate, reference) * reference / (
        np.dot(reference, reference) + eps
    )
    residual = estimate - projection
    return float(
        10.0
        * np.log10(
            (np.dot(projection, projection) + eps)
            / (np.dot(residual, residual) + eps)
        )
    )


def load_utmos(device):
    model = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    ).to(device)
    model.device = device
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-scp", type=Path, required=True)
    parser.add_argument("--enhanced-scp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-utmos", action="store_true",
        help="Only for offline smoke tests; graded evaluation must not use this.",
    )
    parser.add_argument("--invalid-si-sdr", type=float, default=-50.0)
    args = parser.parse_args()
    PENALTY["SI_SDR"] = args.invalid_si_sdr

    references = read_scp(args.reference_scp)
    enhanced = dict(read_scp(args.enhanced_scp))
    device = torch.device(args.device)
    if not args.skip_utmos and device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; pass --device cpu")
    utmos = None if args.skip_utmos else load_utmos(device)

    rows = []
    for uid, reference_path in tqdm(references, desc="Evaluating"):
        values = dict(PENALTY)
        status = "ok"
        try:
            if uid not in enhanced:
                raise FileNotFoundError("enhanced file is missing")
            reference = load_course_wav(reference_path)
            estimate = load_course_wav(enhanced[uid])
            if reference.shape != estimate.shape:
                raise ValueError(
                    f"length mismatch: reference={len(reference)}, enhanced={len(estimate)}"
                )
            values["PESQ"] = float(pesq(16_000, reference, estimate, "wb"))
            values["ESTOI"] = float(stoi(reference, estimate, 16_000, extended=True))
            values["SI_SDR"] = si_sdr(reference, estimate)
            if utmos is not None:
                waveform = torch.from_numpy(estimate).unsqueeze(0).to(device)
                with torch.inference_mode():
                    values["UTMOS"] = float(utmos(waveform, 16_000).cpu().item())
        except Exception as error:
            status = f"penalty: {type(error).__name__}: {error}"
        row = {"id": uid, "status": status, **values}
        if args.skip_utmos:
            row.pop("UTMOS")
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [name for name in PENALTY if not (args.skip_utmos and name == "UTMOS")]
    summary = {
        "num_files": len(rows),
        "num_penalized": sum(row["status"] != "ok" for row in rows),
        "means": {
            name: float(np.mean([float(row[name]) for row in rows]))
            for name in metric_names
        },
        "invalid_file_penalties": {
            name: PENALTY[name] for name in metric_names
        },
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    with (args.output_dir / "RESULTS.txt").open("w", encoding="utf-8") as handle:
        for name, value in summary["means"].items():
            handle.write(f"{name}: {value:.4f}\n")
        handle.write(f"penalized_files: {summary['num_penalized']}\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
