#!/usr/bin/env python3
"""Enhance an unlabeled directory of mono 16-kHz WAV files."""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate import load_model, load_validation_wav  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    files = sorted(args.input_dir.rglob("*.wav"))
    if not files:
        raise ValueError(f"No WAV files found: {args.input_dir}")
    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    for path in files:
        audio = load_validation_wav(path)
        waveform = torch.from_numpy(audio).view(1, -1).to(device)
        lengths = torch.tensor([len(audio)], device=device)
        with torch.inference_mode():
            enhanced, _ = model(waveform, lengths, 16000)
        output = enhanced[0, :len(audio)].float().cpu().numpy()
        if not np.isfinite(output).all():
            raise ValueError(f"Non-finite model output for {path}")
        target = args.output_dir / path.relative_to(args.input_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, output, 16000, subtype="PCM_16")
    print(f"Enhanced {len(files)} files into {args.output_dir}")


if __name__ == "__main__":
    main()
