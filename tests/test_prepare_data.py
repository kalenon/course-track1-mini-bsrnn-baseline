"""Smoke-test the replacement training corpora without external mounts."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data", SCRIPT)
prepare_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_data)


def write_audio(path: Path, sample_rate: int = 16_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate)


def test_libritts_wham_dns_manifest_pipeline(tmp_path, monkeypatch):
    libritts = tmp_path / "libri" / "LibriTTS"
    wham = tmp_path / "WHAM48kHz"
    slr26 = tmp_path / "DNS" / "SLR26"
    slr28 = tmp_path / "DNS" / "SLR28"
    output = tmp_path / "manifests"

    for split in ("train-clean-100", "train-other-500", "dev-clean"):
        write_audio(libritts / split / "speaker" / f"{split}.wav", 24_000)
    for split in ("tr", "cv"):
        write_audio(wham / "wham_noise" / split / f"{split}.wav", 48_000)
    for index in range(10):
        root = slr26 if index < 5 else slr28
        write_audio(root / f"rir_{index:02d}.wav", 48_000)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--libritts-root",
            str(libritts.parent),
            "--wham-root",
            str(wham),
            "--slr26-root",
            str(slr26),
            "--slr28-root",
            str(slr28),
            "--output-dir",
            str(output),
            "--validation-size",
            "1",
        ],
    )
    prepare_data.main()

    statement = json.loads((output / "data_statement.json").read_text())
    counts = {name: data["utterances"] for name, data in statement["manifests"].items()}
    assert counts == {
        "train_clean": 2,
        "valid_clean": 1,
        "train_noise": 1,
        "valid_noise": 1,
        "train_rir": 9,
        "valid_rir": 1,
    }
    assert statement["speech"]["train_sources"] == ["train-clean-100", "train-other-500"]
    assert statement["speech"]["validation_source"] == ["dev-clean"]

    def manifest_paths(name):
        return {
            line.split(maxsplit=1)[1]
            for line in (output / f"{name}.scp").read_text().splitlines()
        }

    assert manifest_paths("train_clean").isdisjoint(manifest_paths("valid_clean"))
    assert manifest_paths("train_noise").isdisjoint(manifest_paths("valid_noise"))
    assert manifest_paths("train_rir").isdisjoint(manifest_paths("valid_rir"))
