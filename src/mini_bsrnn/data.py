"""Manifest-based dynamic mixing for the course Mini-BSRNN baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from scipy.signal import fftconvolve, resample_poly
from torch.utils.data import Dataset, get_worker_info


SAMPLE_RATE = 16_000


def read_scp(path: str | Path) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                uid, audio_path = line.split(maxsplit=1)
            except ValueError as error:
                raise ValueError(f"Malformed line {line_number} in {path}") from error
            entries.append((uid, Path(audio_path).expanduser()))
    if not entries:
        raise ValueError(f"No entries found in {path}")
    return entries


def load_audio(path: str | Path, target_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        audio = resample_poly(
            audio, target_rate // divisor, sample_rate // divisor
        ).astype(np.float32)
    if not np.isfinite(audio).all():
        raise ValueError(f"Non-finite samples in {path}")
    return audio


def crop_or_pad(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) >= length:
        start = int(rng.integers(0, len(audio) - length + 1))
        return audio[start : start + length].copy()
    return np.pad(audio, (0, length - len(audio))).astype(np.float32)


def noise_segment(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if len(audio) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(audio) < length:
        audio = np.tile(audio, math.ceil(length / len(audio)))
    return crop_or_pad(audio, length, rng)


def rms(audio: np.ndarray, eps: float = 1.0e-8) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + eps))


def apply_rir(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(rir)))
    if peak <= 1.0e-8:
        return clean.copy()
    nonzero = np.flatnonzero(np.abs(rir) >= 0.1 * peak)
    if len(nonzero):
        rir = rir[int(nonzero[0]) :]
    rir = rir / (np.sqrt(np.sum(np.square(rir), dtype=np.float64)) + 1.0e-8)
    reverberant = fftconvolve(clean, rir, mode="full")[: len(clean)].astype(np.float32)
    reverberant *= rms(clean) / rms(reverberant)
    return reverberant


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = noise - float(np.mean(noise))
    scale = rms(clean) / (rms(noise) * (10.0 ** (snr_db / 20.0)))
    return (clean + scale * noise).astype(np.float32)


def bandwidth_limit(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Down/up sampling simulates a random telephone or narrow-band channel.
    intermediate_rate = int(rng.choice([8_000, 10_000, 12_000]))
    down = resample_poly(audio, intermediate_rate, SAMPLE_RATE)
    up = resample_poly(down, SAMPLE_RATE, intermediate_rate)
    return up[: len(audio)].astype(np.float32)


def clip_audio(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    threshold = float(np.quantile(np.abs(audio), rng.uniform(0.90, 0.99)))
    return np.clip(audio, -threshold, threshold).astype(np.float32)


def packet_loss(audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = audio.copy()
    frame_size = int(0.020 * SAMPLE_RATE)
    frame_count = math.ceil(len(result) / frame_size)
    lost_frames = max(1, int(frame_count * rng.uniform(0.05, 0.25)))
    remaining = lost_frames
    while remaining > 0:
        run = min(remaining, int(rng.integers(1, 11)))
        start = int(rng.integers(0, max(1, frame_count - run + 1)))
        result[start * frame_size : min(len(result), (start + run) * frame_size)] = 0
        remaining -= run
    return result


class DynamicMixDataset(Dataset):
    """Create one fixed-length degraded mixture per clean utterance."""

    def __init__(
        self,
        clean_scp: str | Path,
        noise_scp: str | Path,
        rir_scp: str | Path,
        segment_samples: int = 32_000,
        seed: int = 2024,
        deterministic: bool = False,
        reverberation_probability: float = 0.5,
        snr_db: Iterable[float] = (-5.0, 20.0),
        bandwidth_probability: float = 0.25,
        clipping_probability: float = 0.25,
        packet_loss_probability: float = 0.25,
    ) -> None:
        self.clean = read_scp(clean_scp)
        self.noise = read_scp(noise_scp)
        self.rirs = read_scp(rir_scp)
        self.segment_samples = int(segment_samples)
        self.seed = int(seed)
        self.deterministic = deterministic
        self.reverb_p = float(reverberation_probability)
        self.snr_low, self.snr_high = (float(value) for value in snr_db)
        self.bandwidth_p = float(bandwidth_probability)
        self.clipping_p = float(clipping_probability)
        self.packet_loss_p = float(packet_loss_probability)

    def __len__(self) -> int:
        return len(self.clean)

    def _rng(self, index: int) -> np.random.Generator:
        if self.deterministic:
            return np.random.default_rng(self.seed + index)
        worker = get_worker_info()
        worker_seed = torch.initial_seed() if worker is None else worker.seed
        return np.random.default_rng((worker_seed + index) % (2**32))

    def __getitem__(self, index: int):
        rng = self._rng(index)
        uid, clean_path = self.clean[index]
        clean = crop_or_pad(load_audio(clean_path), self.segment_samples, rng)

        speech_image = clean.copy()
        if rng.random() < self.reverb_p:
            _, rir_path = self.rirs[int(rng.integers(len(self.rirs)))]
            speech_image = apply_rir(speech_image, load_audio(rir_path))

        _, noise_path = self.noise[int(rng.integers(len(self.noise)))]
        noise = noise_segment(load_audio(noise_path), self.segment_samples, rng)
        mixture = mix_at_snr(speech_image, noise, rng.uniform(self.snr_low, self.snr_high))

        if rng.random() < self.bandwidth_p:
            mixture = bandwidth_limit(mixture, rng)
        if rng.random() < self.clipping_p:
            mixture = clip_audio(mixture, rng)
        if rng.random() < self.packet_loss_p:
            mixture = packet_loss(mixture, rng)

        peak = max(float(np.max(np.abs(clean))), float(np.max(np.abs(mixture))), 1.0)
        clean = (clean / peak).astype(np.float32)
        mixture = (mixture / peak).astype(np.float32)
        return {
            "id": uid,
            "clean": torch.from_numpy(clean),
            "noisy": torch.from_numpy(mixture),
            "length": self.segment_samples,
        }
