"""Force balance E1–E5 (PRD §6.1)."""

from __future__ import annotations

import numpy as np

from src.vehicles import PhysicsConstants, VehicleParams


def traction_forces(
    v_mps: np.ndarray,
    a_mps2: np.ndarray,
    theta_rad: np.ndarray,
    vehicle: VehicleParams,
    constants: PhysicsConstants,
) -> dict[str, np.ndarray]:
    v = np.asarray(v_mps, dtype=float)
    a = np.asarray(a_mps2, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    m = vehicle.mass_kg
    g = constants.g
    # (E1)
    f_aero = 0.5 * constants.rho * vehicle.drag_coefficient * vehicle.frontal_area_m2 * v**2
    # (E2)
    f_roll = m * g * vehicle.rolling_resistance * np.cos(theta)
    # (E3)
    f_grade = m * g * np.sin(theta)
    # (E4)
    f_accel = vehicle.lambda_rot * m * a
    # (E5)
    f_trac = f_aero + f_roll + f_grade + f_accel
    return {
        "f_aero_n": f_aero,
        "f_roll_n": f_roll,
        "f_grade_n": f_grade,
        "f_accel_n": f_accel,
        "f_trac_n": f_trac,
    }
