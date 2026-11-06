"""L2 residual on a physically consistent trajectory (PRD §12.3, §22.2)."""

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.losses.physics_loss import (
    PhysicsInformedLoss,
    consistency_residual,
    consistency_residual_torch,
)
from src.vehicles import KWH_TO_J


def test_consistency_residual_vanishes_on_consistent_soc_power():
    dt = 1.0
    e_usable = 60.0 * KWH_TO_J
    n = 200
    p_hat = np.linspace(5e3, 25e3, n)
    soc = np.empty(n)
    soc[0] = 0.8
    for t in range(n - 1):
        soc[t + 1] = soc[t] - p_hat[t] * dt / e_usable
    r = consistency_residual(soc, p_hat, dt, e_usable)
    assert float(np.max(np.abs(r))) < 1e-8


def test_denormalised_pytorch_residual_pipeline_vanishes():
    """PRD §12.3 / §22.2: Normalise a consistent trajectory, de-normalise in PyTorch, and verify L2 < 1e-8."""
    dt = 1.0
    e_usable = 60.0 * KWH_TO_J
    b, t_seq = 4, 60

    p_raw = np.linspace(4e3, 30e3, b * t_seq, dtype=np.float64).reshape(b, t_seq)
    soc_raw = np.empty((b, t_seq), dtype=np.float64)
    soc_raw[:, 0] = 0.85
    for t in range(t_seq - 1):
        soc_raw[:, t + 1] = soc_raw[:, t] - p_raw[:, t] * dt / e_usable
    rng_raw = np.linspace(250e3, 180e3, b * t_seq, dtype=np.float64).reshape(b, t_seq)

    soc_sc = StandardScaler().fit(soc_raw.reshape(-1, 1))
    pow_sc = StandardScaler().fit(p_raw.reshape(-1, 1))
    rng_sc = StandardScaler().fit(rng_raw.reshape(-1, 1))

    soc_norm = torch.tensor(
        soc_sc.transform(soc_raw.reshape(-1, 1)).reshape(b, t_seq), dtype=torch.float64
    )
    pow_norm = torch.tensor(
        pow_sc.transform(p_raw.reshape(-1, 1)).reshape(b, t_seq), dtype=torch.float64
    )
    rng_norm = torch.tensor(
        rng_sc.transform(rng_raw.reshape(-1, 1)).reshape(b, t_seq), dtype=torch.float64
    )

    # Invert scaler in PyTorch exactly as EVTrajectoryLSTM.forward() does
    soc_hat = soc_norm * float(soc_sc.scale_[0]) + float(soc_sc.mean_[0])
    p_hat_w = pow_norm * float(pow_sc.scale_[0]) + float(pow_sc.mean_[0])
    range_hat_m = rng_norm * float(rng_sc.scale_[0]) + float(rng_sc.mean_[0])

    dt_tensor = torch.full((b,), dt, dtype=torch.float64)
    eu_tensor = torch.full((b,), e_usable, dtype=torch.float64)

    res = consistency_residual_torch(soc_hat, p_hat_w, dt_tensor, eu_tensor)
    assert float(torch.max(torch.abs(res)).item()) < 1e-8

    # Verify PhysicsInformedLoss L2 is independent of corrupted ground-truth labels
    loss_fn = PhysicsInformedLoss(w_soc=1.0, w_pow=1e-8, w_rng=1e-10, lambda_max=1.0, warmup_epochs=1)
    preds = {"soc_hat": soc_hat, "p_hat_w": p_hat_w, "range_hat_m": range_hat_m}
    corrupted_batch = {
        "soc": torch.zeros_like(soc_hat),
        "p_batt_w": torch.full_like(p_hat_w, -99999.0),
        "range_b_m": torch.zeros_like(range_hat_m),
        "soc_mask": torch.ones_like(soc_hat),
        "range_mask": torch.ones_like(range_hat_m),
        "dt_s": dt_tensor,
        "e_usable_j": eu_tensor,
    }
    out = loss_fn(preds, corrupted_batch, epoch=10)
    assert float(out["loss_consist"].item()) < 1e-16
    assert float(out["loss_bounds"].item()) < 1e-16
