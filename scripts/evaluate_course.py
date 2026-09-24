#!/usr/bin/env python3
"""Compute the four current Track 1 objective metrics on paired 16-kHz WAVs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi

PENALTY = {"PESQ": -0.5, "ESTOI": 0.0, "DNSMOS_OVRL": 1.0, "UTMOS": 1.0}


def load(path):
    info = sf.info(path)
    if info.format != "WAV" or info.samplerate != 16000 or info.channels != 1:
        raise ValueError(f"Expected mono 16-kHz WAV: {path}")
    signal, _ = sf.read(path, dtype="float32")
    if not np.isfinite(signal).all():
        raise ValueError(f"Non-finite samples: {path}")
    return signal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--enhanced-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dnsmos-primary", type=Path)
    parser.add_argument("--dnsmos-p808", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-nonintrusive", action="store_true", help="Smoke tests only")
    args = parser.parse_args()
    references = sorted(args.reference_dir.rglob("*.wav"))
    if not references:
        raise ValueError("No reference WAV files")
    device = torch.device(args.device)
    dnsmos = utmos = None
    if not args.skip_nonintrusive:
        if args.dnsmos_primary is None or args.dnsmos_p808 is None:
            parser.error("Both --dnsmos-primary and --dnsmos-p808 are required")
        from espnet2.enh.layers.dnsmos import DNSMOS_local
        dnsmos = DNSMOS_local(str(args.dnsmos_primary), str(args.dnsmos_p808),
                              use_gpu=device.type == "cuda", convert_to_torch=False)
        utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
        utmos.device = device
    rows = []
    for reference_path in references:
        relative = reference_path.relative_to(args.reference_dir)
        values = {"file": str(relative), "status": "ok", **PENALTY}
        try:
            ref = load(reference_path)
            est = load(args.enhanced_dir / relative)
            if len(ref) != len(est):
                raise ValueError("Length mismatch")
            values["PESQ"] = float(pesq(16000, ref, est, "wb"))
            values["ESTOI"] = float(stoi(ref, est, 16000, extended=True))
            if dnsmos is not None:
                values["DNSMOS_OVRL"] = float(dnsmos(est, 16000)["OVRL"])
                with torch.inference_mode():
                    values["UTMOS"] = float(utmos(torch.from_numpy(est).unsqueeze(0).to(device), 16000).cpu().item())
        except Exception as error:
            values.update(PENALTY)
            values["status"] = f"penalty: {type(error).__name__}: {error}"
        if args.skip_nonintrusive:
            values.pop("DNSMOS_OVRL")
            values.pop("UTMOS")
        rows.append(values)
    keys = [key for key in PENALTY if not (args.skip_nonintrusive and key in {"DNSMOS_OVRL", "UTMOS"})]
    summary = {"files": len(rows), "penalized": sum(row["status"] != "ok" for row in rows),
               "means": {key: float(np.mean([row[key] for row in rows])) for key in keys},
               "nonintrusive_skipped": bool(args.skip_nonintrusive)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_file.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
