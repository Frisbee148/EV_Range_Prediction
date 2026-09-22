"""Wheel and battery power E6–E10 (PRD §6.2–§6.3)."""

from __future__ import annotations

import numpy as np

from src.vehicles import PhysicsConstants, VehicleParams


def auxiliary_power(
    t_amb_c: np.ndarray,
    vehicle: VehicleParams,
    constants: PhysicsConstants,
) -> np.ndarray:
    t_amb = np.asarray(t_amb_c, dtype=float)
    # (E10)
    return vehicle.p_aux_base_w + constants.k_hvac_w_per_c * np.abs(t_amb - constants.t_comfort_c)


def wheel_power(f_trac_n: np.ndarray, v_mps: np.ndarray) -> np.ndarray:
    # (E6)
    return np.asarray(f_trac_n, dtype=float) * np.asarray(v_mps, dtype=float)


def battery_power(
    p_wheel_w: np.ndarray,
    v_mps: np.ndarray,
    p_aux_w: np.ndarray,
    vehicle: VehicleParams,
    constants: PhysicsConstants,
) -> np.ndarray:
    p_wheel = np.asarray(p_wheel_w, dtype=float)
    v = np.asarray(v_mps, dtype=float)
    p_aux = np.asarray(p_aux_w, dtype=float)
    p_batt = np.empty_like(p_wheel)
    propel = p_wheel >= 0.0
    regen = ~propel
    above_cutin = v >= constants.v_regen_min_mps
    # (E7)
    p_batt[propel] = p_wheel[propel] / vehicle.eta_dt + p_aux[propel]
    # (E8)
    mask_e8 = regen & above_cutin
    recovered = np.maximum(p_wheel[mask_e8] * vehicle.eta_rg, -vehicle.p_regen_max_w)
    p_batt[mask_e8] = recovered + p_aux[mask_e8]
    # (E9)
    mask_e9 = regen & ~above_cutin
    p_batt[mask_e9] = p_aux[mask_e9]
    return p_batt
