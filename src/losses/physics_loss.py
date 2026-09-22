"""Physics consistency residual L2 on de-normalised outputs (PRD §12.2–§12.3)."""

from __future__ import annotations

import numpy as np


def consistency_residual(
    soc_hat: np.ndarray,
    p_hat_w: np.ndarray,
    dt_s: float,
    e_usable_j: float,
) -> np.ndarray:
    """r[t] = SOC_hat[t+1] - (SOC_hat[t] - P_hat[t] * dt / E_usable)."""
    soc = np.asarray(soc_hat, dtype=float)
    p = np.asarray(p_hat_w, dtype=float)
    return soc[1:] - (soc[:-1] - p[:-1] * dt_s / e_usable_j)
