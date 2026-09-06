#!/usr/bin/env python3
"""Report parameters and reproducible learned-layer MAC/s for Mini-BSRNN."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mini_bsrnn.model import HOP_LENGTH, SUBBANDS, MiniBSRNN_SE  # noqa: E402


def learned_layer_macs(samples: int, embedding: int = 64, layers: int = 2):
    frames = samples // HOP_LENGTH + 1  # torch.stft(center=True)
    bands = len(SUBBANDS)
    bins = sum(SUBBANDS)

    band_split = frames * embedding * (2 * bins)
    # Each axis: bidirectional LSTM with hidden=2E plus a 4E -> E projection.
    recurrent_per_axis = frames * bands * (48 * embedding**2)
    projection_per_axis = frames * bands * (4 * embedding**2)
    recurrent_stack = layers * 2 * (recurrent_per_axis + projection_per_axis)
    # Mask and residual heads: E->4E and 4E->4*band_width for every band.
    decoder = 2 * frames * (
        bands * 4 * embedding**2 + 16 * embedding * bins
    )
    return {
        "band_split": band_split,
        "recurrent_stack": recurrent_stack,
        "decoder": decoder,
        "total": band_split + recurrent_stack + decoder,
        "stft_frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path,
        default=Path("logs/metrics_and_complexity/complexity.json"),
    )
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    model = MiniBSRNN_SE(embedding_dim=64, num_layers=2)
    samples = round(16_000 * args.duration)
    macs = learned_layer_macs(samples)
    report = {
        "protocol": {
            "batch_size": 1,
            "channels": 1,
            "sample_rate": 16000,
            "duration_seconds": args.duration,
            "definition": (
                "multiply-accumulates in learned Conv1d, Linear and LSTM layers; "
                "STFT/ISTFT, normalization and pointwise operations are reported as excluded"
            ),
        },
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_millions": sum(parameter.numel() for parameter in model.parameters()) / 1.0e6,
        "learned_layer_macs": macs,
        "gmac_per_second": macs["total"] / args.duration / 1.0e9,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

