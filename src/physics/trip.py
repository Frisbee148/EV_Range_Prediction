"""Kinematics helpers and full E1–E14 trip simulation."""

from __future__ import annotations

import numpy as np

from src.physics.battery_model import integrate_soc, remaining_energy
from src.physics.power_model import auxiliary_power, battery_power, wheel_power
from src.physics.range_model import range_definition_a, range_definition_b
from src.physics.vehicle_dynamics import traction_forces
from src.vehicles import PhysicsConstants, VehicleParams


def kinematics_from_speed_elevation(
    v_mps: np.ndarray,
    elevation_m: np.ndarray,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(v_mps, dtype=float)
    z = np.asarray(elevation_m, dtype=float)
    a = np.gradient(v, dt_s)
    dx = np.maximum(v * dt_s, 1e-9)
    dz = np.empty_like(z)
    dz[0] = 0.0
    dz[1:] = np.diff(z)
    theta = np.arctan2(dz, dx)
    return a, theta


def simulate_trip(
    v_mps: np.ndarray,
    a_mps2: np.ndarray,
    theta_rad: np.ndarray,
    t_amb_c: np.ndarray,
    soc_0: float,
    dt_s: float,
    vehicle: VehicleParams,
    constants: PhysicsConstants,
) -> dict[str, np.ndarray | bool]:
    forces = traction_forces(v_mps, a_mps2, theta_rad, vehicle, constants)
    p_aux = auxiliary_power(t_amb_c, vehicle, constants)
    p_wheel = wheel_power(forces["f_trac_n"], v_mps)
    p_batt = battery_power(p_wheel, v_mps, p_aux, vehicle, constants)
    soc = integrate_soc(p_batt, soc_0, dt_s, vehicle)
    e_rem = remaining_energy(soc, vehicle)
    r_a, r_a_mask = range_definition_a(p_batt, v_mps, e_rem, dt_s, constants)
    r_b, r_b_ok = range_definition_b(p_batt, v_mps, e_rem, dt_s, vehicle)
    out: dict[str, np.ndarray | bool] = {
        **forces,
        "p_aux_w": p_aux,
        "p_wheel_w": p_wheel,
        "p_batt_w": p_batt,
        "soc": soc,
        "e_rem_j": e_rem,
        "range_a_m": r_a,
        "range_a_mask": r_a_mask,
        "range_b_m": r_b,
        "range_b_defined": r_b_ok,
    }
    return out
