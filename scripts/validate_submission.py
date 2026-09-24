#!/usr/bin/env python3
"""Check blind-test enhancement files without reference clean speech."""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = {p.relative_to(args.input_dir) for p in args.input_dir.rglob("*.wav")}
    outputs = {p.relative_to(args.output_dir) for p in args.output_dir.rglob("*.wav")}
    if not inputs or inputs != outputs:
        raise ValueError(f"Input/output WAV sets differ; missing={sorted(inputs-outputs)}, extra={sorted(outputs-inputs)}")
    for relative in sorted(inputs):
        source = sf.info(args.input_dir / relative)
        target = sf.info(args.output_dir / relative)
        if source.samplerate != 16000 or source.channels != 1 or target.samplerate != 16000 or target.channels != 1 or source.frames != target.frames:
            raise ValueError(f"Format or length mismatch: {relative}")
        signal, _ = sf.read(args.output_dir / relative, dtype="float32")
        if not np.isfinite(signal).all():
            raise ValueError(f"Non-finite output: {relative}")
    print(f"Valid enhancement submission: {len(inputs)} files")


if __name__ == "__main__":
    main()
