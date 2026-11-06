"""Physics consistency residual (L2), bounds/monotonicity (L3), range consistency (L4),
and composite Physics-Informed loss (L5-L6) on de-normalised outputs (PRD §12.2-§12.3).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.data_loss import DataLoss


def consistency_residual(
    soc_hat: np.ndarray,
    p_hat_w: np.ndarray,
    dt_s: float,
    e_usable_j: float,
) -> np.ndarray:
    """r[t] = SOC_hat[t+1] - (SOC_hat[t] - P_hat[t] * dt / E_usable) (NumPy)."""
    soc = np.asarray(soc_hat, dtype=float)
    p = np.asarray(p_hat_w, dtype=float)
    return soc[..., 1:] - (soc[..., :-1] - p[..., :-1] * dt_s / e_usable_j)


def consistency_residual_torch(
    soc_hat: torch.Tensor,
    p_hat_w: torch.Tensor,
    dt_s: torch.Tensor | float,
    e_usable_j: torch.Tensor | float,
) -> torch.Tensor:
    """(L2) Physics consistency residual on de-normalised PyTorch tensors.

    Uses NO ground-truth labels: `soc_hat` (dimensionless [0, 1]) and `p_hat_w` (Watts)
    are both model outputs; `dt_s` (seconds) and `e_usable_j` (Joules) are constants.
    Supports shapes `[T_seq]` or `[B, T_seq]`.
    """
    if isinstance(dt_s, torch.Tensor) and dt_s.ndim == 1 and soc_hat.ndim == 2:
        dt = dt_s.unsqueeze(-1).to(dtype=soc_hat.dtype, device=soc_hat.device)
    else:
        dt = dt_s

    if isinstance(e_usable_j, torch.Tensor) and e_usable_j.ndim == 1 and soc_hat.ndim == 2:
        e_u = e_usable_j.unsqueeze(-1).to(dtype=soc_hat.dtype, device=soc_hat.device)
    else:
        e_u = e_usable_j

    delta_soc = p_hat_w[..., :-1] * dt / e_u
    return soc_hat[..., 1:] - (soc_hat[..., :-1] - delta_soc)


def bounds_and_monotonicity_loss(
    soc_hat: torch.Tensor,
    p_hat_w: torch.Tensor,
    range_hat_m: torch.Tensor,
    w_soc: torch.Tensor | float = 1.0,
    w_rng: torch.Tensor | float = 1.0,
) -> dict[str, torch.Tensor]:
    """(L3) Bound and monotonicity penalties on de-normalised predictions (PRD §12.2)."""
    t_scale = float(soc_hat.shape[-1] ** 2) if soc_hat.ndim >= 1 else 1.0
    soc_bound = torch.mean(F.relu(-soc_hat) ** 2 + F.relu(soc_hat - 1.0) ** 2) * w_soc

    dsoc = soc_hat[..., 1:] - soc_hat[..., :-1]
    discharging_mask = (p_hat_w[..., :-1] > 0.0).to(dtype=soc_hat.dtype)
    mono_viol = (F.relu(dsoc) ** 2) * discharging_mask
    soc_mono = torch.mean(mono_viol) * w_soc * t_scale

    rng_neg = torch.mean(F.relu(-range_hat_m) ** 2) * w_rng

    l_bounds = soc_bound + soc_mono + rng_neg
    return {
        "loss_soc_bound": soc_bound,
        "loss_soc_mono": soc_mono,
        "loss_rng_neg": rng_neg,
        "loss_bounds": l_bounds,
    }


def range_consistency_loss(
    soc_hat: torch.Tensor,
    p_hat_w: torch.Tensor,
    range_hat_m: torch.Tensor,
    speed_mps_raw: torch.Tensor,
    e_usable_j: torch.Tensor | float,
    w_rng: torch.Tensor | float = 1.0,
    v_min_mps: float = 0.5,
    ema_alpha: float = 0.05,
    eps: float = 1.0,
) -> torch.Tensor:
    """(L4) Optional consistency between range head and model's own SOC/power (PRD §12.2)."""
    if isinstance(e_usable_j, torch.Tensor) and e_usable_j.ndim == 1 and soc_hat.ndim == 2:
        e_u = e_usable_j.unsqueeze(-1).to(dtype=soc_hat.dtype, device=soc_hat.device)
    else:
        e_u = e_usable_j

    v_safe = torch.clamp(speed_mps_raw.to(dtype=soc_hat.dtype, device=soc_hat.device), min=v_min_mps)
    inst_rate = p_hat_w / v_safe  # J/m

    # Causal EMA along time dimension
    e_bar_steps = []
    cur = inst_rate[..., 0]
    for t in range(inst_rate.shape[-1]):
        cur = ema_alpha * inst_rate[..., t] + (1.0 - ema_alpha) * cur
        e_bar_steps.append(cur)
    e_bar_hat = torch.stack(e_bar_steps, dim=-1)

    denom = torch.clamp(e_bar_hat, min=eps)
    r_implied = torch.clamp(soc_hat, min=0.0, max=1.0) * e_u / denom
    return torch.mean((range_hat_m - r_implied) ** 2) * w_rng


def lambda_schedule(epoch: int, warmup_epochs: int, lambda_max: float) -> float:
    """Linear lambda warm-up: `lambda(epoch) = lambda_max * min(1.0, epoch / warmup_epochs)`."""
    if lambda_max <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return float(lambda_max)
    return float(lambda_max) * min(1.0, max(0.0, float(epoch) / float(warmup_epochs)))


class PhysicsInformedLoss(nn.Module):
    """Combined data + physics consistency loss (L1-L6) with de-normalisation support (PRD §12.2-§12.3)."""

    def __init__(
        self,
        w_soc: float = 1.0,
        w_pow: float = 1.0,
        w_rng: float = 1.0,
        lambda_max: float = 1.0,
        warmup_epochs: int = 10,
        use_range_consistency: bool = False,
        c_r: float = 0.1,
    ) -> None:
        super().__init__()
        self.data_loss = DataLoss(w_soc=w_soc, w_pow=w_pow, w_rng=w_rng)
        self.lambda_max = float(lambda_max)
        self.warmup_epochs = int(warmup_epochs)
        self.use_range_consistency = bool(use_range_consistency)
        self.c_r = float(c_r)

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> dict[str, torch.Tensor]:
        """Compute L_data, L_consist, L_bounds, L_physics, and L_total.

        `predictions` must provide de-normalised physical SI quantities:
        - `soc_hat`: dimensionless [0, 1]
        - `p_hat_w`: Watts (W)
        - `range_hat_m`: meters (m)
        """
        soc_hat = predictions["soc_hat"]
        p_hat_w = predictions["p_hat_w"]
        range_hat_m = predictions["range_hat_m"]

        d_losses = self.data_loss(
            soc_hat=soc_hat,
            p_hat_w=p_hat_w,
            range_hat_m=range_hat_m,
            soc_true=batch["soc"],
            p_true_w=batch["p_batt_w"],
            range_true_m=batch["range_b_m"],
            soc_label_mask=batch.get("soc_mask", 1.0),
            range_valid_mask=batch.get("range_mask", 1.0),
        )

        # (L2) Consistency residual on de-normalised predictions (uses NO ground-truth labels)
        residual = consistency_residual_torch(
            soc_hat=soc_hat,
            p_hat_w=p_hat_w,
            dt_s=batch["dt_s"],
            e_usable_j=batch["e_usable_j"],
        )
        raw_consist_mse = torch.mean(residual ** 2)
        t_scale = float(soc_hat.shape[-1] ** 2) if soc_hat.ndim >= 1 else 1.0
        l_consist = raw_consist_mse * self.data_loss.w_soc * t_scale

        # (L3) Bounds and monotonicity penalties
        b_losses = bounds_and_monotonicity_loss(
            soc_hat=soc_hat,
            p_hat_w=p_hat_w,
            range_hat_m=range_hat_m,
            w_soc=self.data_loss.w_soc,
            w_rng=self.data_loss.w_rng,
        )
        l_bounds = b_losses["loss_bounds"]

        # (L4) Optional range consistency
        if self.use_range_consistency and "speed_mps_raw" in batch:
            l_range_c = range_consistency_loss(
                soc_hat=soc_hat,
                p_hat_w=p_hat_w,
                range_hat_m=range_hat_m,
                speed_mps_raw=batch["speed_mps_raw"],
                e_usable_j=batch["e_usable_j"],
                w_rng=self.data_loss.w_rng,
            )
        else:
            l_range_c = torch.zeros((), dtype=soc_hat.dtype, device=soc_hat.device)

        # (L5) Total physics penalty
        l_physics = l_consist + l_bounds + self.c_r * l_range_c

        # (L6) Total loss with warm-up schedule
        lam = lambda_schedule(epoch=epoch, warmup_epochs=self.warmup_epochs, lambda_max=self.lambda_max)
        l_total = d_losses["loss_data"] + lam * l_physics

        return {
            **d_losses,
            **b_losses,
            "loss_consist": l_consist,
            "loss_range_c": l_range_c,
            "loss_physics": l_physics,
            "lambda_t": torch.tensor(lam, dtype=soc_hat.dtype, device=soc_hat.device),
            "loss_total": l_total,
            "residual_rms": torch.sqrt(torch.clamp(raw_consist_mse.detach(), min=0.0)),
        }
