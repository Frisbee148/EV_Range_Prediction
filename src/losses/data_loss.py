"""Supervised multi-head data loss (L0) and (L1) with masking (PRD §12.2)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_head_weights(
    soc_train: np.ndarray,
    p_batt_train: np.ndarray,
    range_b_train: np.ndarray,
    range_valid_mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute (L0) inverse-variance weights on the training split only (PRD §12.2)."""
    var_soc = float(np.var(np.asarray(soc_train, dtype=np.float64)))
    var_pow = float(np.var(np.asarray(p_batt_train, dtype=np.float64)))
    rng = np.asarray(range_b_train, dtype=np.float64)
    if range_valid_mask is not None:
        mask = np.asarray(range_valid_mask, dtype=bool)
        var_rng = float(np.var(rng[mask])) if np.any(mask) else float(np.var(rng))
    else:
        var_rng = float(np.var(rng))
    return {
        "w_soc": 1.0 / max(var_soc, eps),
        "w_pow": 1.0 / max(var_pow, eps),
        "w_rng": 1.0 / max(var_rng, eps),
    }


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | float | int = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean squared error over masked entries."""
    sq_err = (pred - target) ** 2
    if isinstance(mask, (int, float)):
        if float(mask) == 1.0:
            return torch.mean(sq_err)
        return torch.mean(sq_err) * float(mask)
    m = mask.to(dtype=sq_err.dtype, device=sq_err.device)
    denom = torch.clamp(torch.sum(m), min=eps)
    return torch.sum(sq_err * m) / denom


class DataLoss(nn.Module):
    """(L1) Supervised data loss with (L0) per-head inverse-variance normalisation."""

    def __init__(
        self,
        w_soc: float = 1.0,
        w_pow: float = 1.0,
        w_rng: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("w_soc", torch.tensor(float(w_soc), dtype=torch.float32))
        self.register_buffer("w_pow", torch.tensor(float(w_pow), dtype=torch.float32))
        self.register_buffer("w_rng", torch.tensor(float(w_rng), dtype=torch.float32))

    def forward(
        self,
        soc_hat: torch.Tensor,
        p_hat_w: torch.Tensor,
        range_hat_m: torch.Tensor,
        soc_true: torch.Tensor,
        p_true_w: torch.Tensor,
        range_true_m: torch.Tensor,
        soc_label_mask: torch.Tensor | float = 1.0,
        range_valid_mask: torch.Tensor | float = 1.0,
    ) -> dict[str, torch.Tensor]:
        l_soc = self.w_soc * masked_mse(soc_hat, soc_true, soc_label_mask)
        l_pow = self.w_pow * masked_mse(p_hat_w, p_true_w, 1.0)
        l_rng = self.w_rng * masked_mse(range_hat_m, range_true_m, range_valid_mask)
        l_data = l_soc + l_pow + l_rng
        return {
            "loss_soc": l_soc,
            "loss_pow": l_pow,
            "loss_rng": l_rng,
            "loss_data": l_data,
        }
