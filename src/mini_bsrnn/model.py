"""A compact, fixed-16-kHz BSRNN speech enhancement model.

This module implements a compact course baseline with band splitting,
alternating time/frequency BLSTMs, and complex mask plus residual decoding.
The architecture is deliberately fixed to 16 kHz and excludes frequency bands
above 8 kHz.
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn


SAMPLE_RATE = 16_000
N_FFT = 320
HOP_LENGTH = 160

# 50-Hz STFT resolution: 0--4 kHz uses 200-Hz bands (with DC in
# the first band), and 4--8 kHz uses 500-Hz bands. The 28 bands cover all 161
# one-sided STFT bins exactly.
SUBBANDS = (5,) + (4,) * 19 + (10,) * 8


def _sample_rate_to_int(fs: Union[int, torch.Tensor]) -> int:
    if isinstance(fs, torch.Tensor):
        values = fs.detach().cpu().reshape(-1).tolist()
        if not values or any(int(value) != int(values[0]) for value in values):
            raise ValueError("All examples in a batch must have one sample rate")
        return int(values[0])
    return int(fs)


class BandSplit(nn.Module):
    """Project each complex frequency band to a shared embedding dimension."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.norms = nn.ModuleList()
        self.projections = nn.ModuleList()
        for width in SUBBANDS:
            input_channels = 2 * width
            self.norms.append(nn.GroupNorm(1, input_channels))
            self.projections.append(nn.Conv1d(input_channels, embedding_dim, 1))

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        # spectrum: [batch, frames, frequency], complex
        if not spectrum.is_complex():
            raise TypeError("BandSplit expects a complex STFT tensor")

        bands = []
        start = 0
        real_spectrum = torch.view_as_real(spectrum)
        for width, norm, projection in zip(SUBBANDS, self.norms, self.projections):
            band = real_spectrum[:, :, start : start + width]
            band = band.reshape(band.size(0), band.size(1), 2 * width)
            band = projection(norm(band.transpose(1, 2)))
            bands.append(band)
            start += width
        return torch.stack(bands, dim=-1)  # [batch, embedding, frames, bands]


class MaskResidualDecoder(nn.Module):
    """Decode one complex ratio mask and one complex residual per band."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.mask_heads = nn.ModuleList(
            self._make_head(embedding_dim, width) for width in SUBBANDS
        )
        self.residual_heads = nn.ModuleList(
            self._make_head(embedding_dim, width) for width in SUBBANDS
        )

    @staticmethod
    def _make_head(embedding_dim: int, width: int) -> nn.Sequential:
        return nn.Sequential(
            nn.GroupNorm(1, embedding_dim),
            nn.Conv1d(embedding_dim, 4 * embedding_dim, 1),
            nn.Tanh(),
            nn.Conv1d(4 * embedding_dim, 4 * width, 1),
            nn.GLU(dim=1),
        )

    @staticmethod
    def _decode_head(features: torch.Tensor, head: nn.Module, width: int) -> torch.Tensor:
        decoded = head(features).transpose(1, 2).contiguous()
        return decoded.reshape(decoded.size(0), decoded.size(1), width, 2)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        masks = []
        residuals = []
        for index, width in enumerate(SUBBANDS):
            band_features = features[:, :, :, index]
            masks.append(self._decode_head(band_features, self.mask_heads[index], width))
            residuals.append(
                self._decode_head(band_features, self.residual_heads[index], width)
            )
        mask = torch.view_as_complex(torch.cat(masks, dim=2))
        residual = torch.view_as_complex(torch.cat(residuals, dim=2))
        return mask, residual


class MiniBSRNN(nn.Module):
    """Two-axis recurrent BSRNN core operating on a 16-kHz complex STFT."""

    def __init__(self, embedding_dim: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        if embedding_dim <= 0 or num_layers <= 0:
            raise ValueError("embedding_dim and num_layers must be positive")

        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        recurrent_dim = 2 * embedding_dim

        self.band_split = BandSplit(embedding_dim)
        self.time_norms = nn.ModuleList()
        self.time_rnns = nn.ModuleList()
        self.time_projections = nn.ModuleList()
        self.frequency_norms = nn.ModuleList()
        self.frequency_rnns = nn.ModuleList()
        self.frequency_projections = nn.ModuleList()

        for _ in range(num_layers):
            self.time_norms.append(nn.GroupNorm(1, embedding_dim))
            self.time_rnns.append(
                nn.LSTM(
                    embedding_dim,
                    recurrent_dim,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.time_projections.append(nn.Linear(4 * embedding_dim, embedding_dim))

            self.frequency_norms.append(nn.GroupNorm(1, embedding_dim))
            self.frequency_rnns.append(
                nn.LSTM(
                    embedding_dim,
                    recurrent_dim,
                    batch_first=True,
                    bidirectional=True,
                )
            )
            self.frequency_projections.append(
                nn.Linear(4 * embedding_dim, embedding_dim)
            )

        self.decoder = MaskResidualDecoder(embedding_dim)

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        features = self.band_split(spectrum)
        batch, embedding, frames, bands = features.shape

        for layer in range(self.num_layers):
            time_features = self.time_norms[layer](features)
            time_features = time_features.permute(0, 3, 2, 1).reshape(
                batch * bands, frames, embedding
            )
            time_features, _ = self.time_rnns[layer](time_features)
            time_features = self.time_projections[layer](time_features)
            time_features = time_features.reshape(batch, bands, frames, embedding)
            features = features + time_features.permute(0, 3, 2, 1)

            frequency_features = self.frequency_norms[layer](features)
            frequency_features = frequency_features.permute(0, 2, 3, 1).reshape(
                batch * frames, bands, embedding
            )
            frequency_features, _ = self.frequency_rnns[layer](frequency_features)
            frequency_features = self.frequency_projections[layer](frequency_features)
            frequency_features = frequency_features.reshape(
                batch, frames, bands, embedding
            )
            features = features + frequency_features.permute(0, 3, 1, 2)

        mask, residual = self.decoder(features)
        return mask * spectrum + residual


class MiniBSRNN_SE(nn.Module):
    """Waveform wrapper with a stable enhancement-model return contract."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 2,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
    ) -> None:
        super().__init__()
        if sample_rate != SAMPLE_RATE or n_fft != N_FFT or hop_length != HOP_LENGTH:
            raise ValueError(
                "MiniBSRNN_SE is fixed to sample_rate=16000, n_fft=320, "
                "hop_length=160"
            )
        if sum(SUBBANDS) != n_fft // 2 + 1:
            raise RuntimeError("The configured subbands do not cover the STFT bins")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.bsrnn = MiniBSRNN(embedding_dim=embedding_dim, num_layers=num_layers)

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        fs: Optional[Union[int, torch.Tensor]] = SAMPLE_RATE,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if speech_mix.ndim != 2:
            raise ValueError("speech_mix must have shape [batch, samples]")
        if fs is not None and _sample_rate_to_int(fs) != self.sample_rate:
            raise ValueError(
                f"MiniBSRNN_SE only accepts {self.sample_rate}-Hz audio; got "
                f"{_sample_rate_to_int(fs)} Hz"
            )

        lengths = torch.as_tensor(speech_lengths, device=speech_mix.device).reshape(-1)
        if lengths.numel() != speech_mix.size(0):
            raise ValueError("speech_lengths must contain one value per example")

        window = self.window.to(device=speech_mix.device, dtype=speech_mix.dtype)
        spectrum = torch.stft(
            speech_mix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            return_complex=True,
        ).transpose(1, 2)
        enhanced_spectrum = self.bsrnn(spectrum)
        enhanced_wav = torch.istft(
            enhanced_spectrum.transpose(1, 2),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            center=True,
            length=speech_mix.size(-1),
        )

        valid = torch.arange(speech_mix.size(-1), device=speech_mix.device)
        enhanced_wav = enhanced_wav * (valid.unsqueeze(0) < lengths.unsqueeze(1))
        return enhanced_wav, enhanced_spectrum


if __name__ == "__main__":
    model = MiniBSRNN_SE()
    waveform = torch.randn(1, SAMPLE_RATE)
    length = torch.tensor([waveform.size(-1)])
    enhanced, stft = model(waveform, length, SAMPLE_RATE)
    print(f"waveform={tuple(enhanced.shape)}, stft={tuple(stft.shape)}")
