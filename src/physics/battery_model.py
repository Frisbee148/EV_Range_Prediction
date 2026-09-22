"""SOC integration E11–E12 (PRD §6.4)."""

from __future__ import annotations

import numpy as np

from src.vehicles import VehicleParams


def integrate_soc(
    p_batt_w: np.ndarray,
    soc_0: float,
    dt_s: float,
    vehicle: VehicleParams,
) -> np.ndarray:
    p = np.asarray(p_batt_w, dtype=float)
    n = p.shape[0]
    soc = np.empty(n, dtype=float)
    soc[0] = float(np.clip(soc_0, 0.0, 1.0))
    scale = dt_s / vehicle.e_usable_j
    for t in range(n - 1):
        # (E11)
        soc[t + 1] = float(np.clip(soc[t] - p[t] * scale, 0.0, 1.0))
    return soc


def remaining_energy(soc: np.ndarray, vehicle: VehicleParams) -> np.ndarray:
    # (E12)
    return np.asarray(soc, dtype=float) * vehicle.e_usable_j
