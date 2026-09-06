"""Lightweight losses used by Mini-BSRNN without importing ESPnet."""

import torch
from torch import nn


class MultiResL1SpecLoss(nn.Module):
    def __init__(
        self,
        window_sz=(256, 512, 768, 1024),
        eps=1.0e-6,
        normalize_variance=True,
        time_domain_weight=0.5,
    ):
        super().__init__()
        self.window_sz = tuple(window_sz)
        self.eps = eps
        self.normalize_variance = normalize_variance
        self.time_domain_weight = time_domain_weight

    def forward(self, target, estimate):
        if target.shape != estimate.shape:
            raise ValueError(f"Shape mismatch: {target.shape} != {estimate.shape}")
        if target.dtype in (torch.float16, torch.bfloat16):
            target = target.float()
        if estimate.dtype in (torch.float16, torch.bfloat16):
            estimate = estimate.float()
        if self.normalize_variance:
            # Match ESPnet's MultiResL1SpecLoss.  Adding eps to the variance
            # before the square root gives a smooth, 1/sqrt(eps)-bounded gain
            # for silent and very-low-energy signals.  Clamping the standard
            # deviation itself at eps permits a much larger 1/eps gain and can
            # produce extreme gradients around the clamp boundary.
            target = target / torch.sqrt(
                torch.var(target, dim=1, keepdim=True, unbiased=False) + self.eps
            )
            estimate = estimate / torch.sqrt(
                torch.var(estimate, dim=1, keepdim=True, unbiased=False)
                + self.eps
            )

        scale = (estimate * target).sum(-1, keepdim=True) / (
            estimate.square().sum(-1, keepdim=True) + self.eps
        )
        scaled_estimate = estimate * scale
        time_loss = (scaled_estimate - target).abs().sum(dim=-1)

        spectral_loss = torch.zeros_like(time_loss)
        for window_size in self.window_sz:
            window = torch.ones(
                window_size, device=target.device, dtype=target.dtype
            )
            target_stft = torch.stft(
                target,
                n_fft=window_size,
                hop_length=window_size // 2,
                win_length=window_size,
                window=window,
                center=True,
                return_complex=True,
            )
            estimate_stft = torch.stft(
                scaled_estimate,
                n_fft=window_size,
                hop_length=window_size // 2,
                win_length=window_size,
                window=window,
                center=True,
                return_complex=True,
            )
            spectral_loss += (estimate_stft.abs() - target_stft.abs()).abs().sum(
                dim=(1, 2)
            )
        spectral_loss /= len(self.window_sz)
        return (
            self.time_domain_weight * time_loss
            + (1.0 - self.time_domain_weight) * spectral_loss
        )


class SISNRLoss(nn.Module):
    def __init__(self, eps=1.0e-8):
        super().__init__()
        self.eps = eps

    def forward(self, reference, estimate):
        reference = reference - reference.mean(dim=-1, keepdim=True)
        estimate = estimate - estimate.mean(dim=-1, keepdim=True)
        projection = (
            (estimate * reference).sum(dim=-1, keepdim=True)
            * reference
            / (reference.square().sum(dim=-1, keepdim=True) + self.eps)
        )
        noise = estimate - projection
        ratio = projection.square().sum(dim=-1) / (
            noise.square().sum(dim=-1) + self.eps
        )
        return -10.0 * torch.log10(ratio + self.eps)
