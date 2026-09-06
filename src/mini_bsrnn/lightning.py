"""PyTorch Lightning training wrapper for Mini-BSRNN."""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from .losses import MultiResL1SpecLoss
from .model import MiniBSRNN_SE


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1.0e-8):
    reference = reference - reference.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    projection = (
        (estimate * reference).sum(dim=-1, keepdim=True)
        * reference
        / (reference.square().sum(dim=-1, keepdim=True) + eps)
    )
    residual = estimate - projection
    return 10.0 * torch.log10(
        (projection.square().sum(dim=-1) + eps)
        / (residual.square().sum(dim=-1) + eps)
    )


class MiniBSRNNModule(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 2,
        learning_rate: float = 1.0e-3,
        weight_decay: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.se_model = MiniBSRNN_SE(
            embedding_dim=embedding_dim, num_layers=num_layers
        )
        self.loss_fn = MultiResL1SpecLoss(
            window_sz=(256, 512, 768, 1024),
            eps=1.0e-6,
            normalize_variance=True,
            time_domain_weight=0.5,
        )

    def forward(self, noisy, lengths):
        return self.se_model(noisy, lengths, 16_000)[0]

    def _shared_step(self, batch, stage: str):
        clean = batch["clean"].float()
        noisy = batch["noisy"].float()
        lengths = torch.as_tensor(batch["length"], device=self.device)
        enhanced = self(noisy, lengths)
        loss = self.loss_fn(clean, enhanced).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {stage} loss")
        score = si_sdr(clean, enhanced).mean()
        batch_size = clean.shape[0]
        self.log(
            f"{stage}_loss", loss, prog_bar=True, sync_dist=True,
            on_step=stage == "train", on_epoch=True, batch_size=batch_size,
        )
        self.log(
            f"{stage}_si_sdr", score, prog_bar=True, sync_dist=True,
            on_step=False, on_epoch=True, batch_size=batch_size,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=1.0e-8,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.85
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

