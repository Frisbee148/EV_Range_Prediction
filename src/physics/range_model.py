"""Remaining-range labels E13–E14 (PRD §7)."""

from __future__ import annotations

import numpy as np

from src.vehicles import PhysicsConstants, VehicleParams


def _ema(x: np.ndarray, dt_s: float, window_s: float, init_s: float) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=float)
    n_init = max(1, int(round(init_s / dt_s)))
    out[0] = float(np.mean(x[:n_init]))
    alpha = dt_s / window_s
    for t in range(1, n):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


def range_definition_a(
    p_batt_w: np.ndarray,
    v_mps: np.ndarray,
    e_rem_j: np.ndarray,
    dt_s: float,
    constants: PhysicsConstants,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_batt_w, dtype=float)
    v = np.asarray(v_mps, dtype=float)
    e_rem = np.asarray(e_rem_j, dtype=float)
    inst = p / np.maximum(v, constants.v_min_mps)
    e_bar = _ema(inst, dt_s, constants.ema_window_s, constants.ema_init_s)
    r_a = e_rem / np.maximum(e_bar, constants.eps)
    dx = np.maximum(v, 0.0) * dt_s
    n_win = max(1, int(round(constants.ema_window_s / dt_s)))
    cdist = np.cumsum(dx)
    trailing = cdist.copy()
    if n_win < len(cdist):
        trailing[n_win:] = cdist[n_win:] - cdist[: -n_win]
    valid = (trailing >= constants.d_min_m).astype(int)
    return r_a, valid


def range_definition_b(
    p_batt_w: np.ndarray,
    v_mps: np.ndarray,
    e_rem_j: np.ndarray,
    dt_s: float,
    vehicle: VehicleParams,
) -> tuple[np.ndarray, bool]:
    """Oracle cycle-repeat range. Returns (R_B metres, defined)."""
    p = np.asarray(p_batt_w, dtype=float)
    v = np.asarray(v_mps, dtype=float)
    e_rem = np.asarray(e_rem_j, dtype=float)
    n = p.shape[0]
    de = p * dt_s
    dx = v * dt_s
    net = float(de.sum())
    if net <= 0.0:
        return np.full(n, np.nan), False
    n_laps = int(np.ceil(vehicle.e_usable_j / net)) + 2
    de_t = np.tile(de, n_laps)
    dx_t = np.tile(dx, n_laps)
    ce = np.concatenate(([0.0], np.cumsum(de_t)))
    cx = np.concatenate(([0.0], np.cumsum(dx_t)))
    r_b = np.empty(n, dtype=float)
    total = ce.shape[0]
    for t in range(n):
        target = ce[t] + e_rem[t]
        n_star = int(np.searchsorted(ce, target, side="left"))
        if n_star >= total:
            r_b[t] = np.nan
        else:
            r_b[t] = cx[n_star] - cx[t]
    return r_b, True
