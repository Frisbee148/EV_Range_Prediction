"""Assemble PRD §10 rows from kinematics + physics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.physics.trip import simulate_trip
from src.vehicles import PhysicsConstants, VehicleParams

SCHEMA = (
    "timestamp",
    "vehicle_id",
    "trip_id",
    "scenario",
    "speed_mps",
    "accel_mps2",
    "grade_rad",
    "elevation_m",
    "distance_m",
    "stop_flag",
    "ambient_temp_c",
    "mass_kg",
    "drag_coefficient",
    "frontal_area_m2",
    "rolling_resistance",
    "battery_capacity_j",
    "drivetrain_efficiency",
    "regen_efficiency",
    "aux_power_base_w",
    "rotating_mass_factor",
    "p_wheel_w",
    "p_batt_w",
    "soc",
    "energy_remaining_j",
    "range_a_m",
    "range_b_m",
    "range_valid_mask",
    "soc_label_mask",
    "noise_seed",
)


def labeled_trip(
    kin: dict[str, np.ndarray],
    vehicle: VehicleParams,
    constants: PhysicsConstants,
    *,
    trip_id: str,
    scenario: str,
    dt_s: float,
    ambient_temp_c: float,
    soc_0: float,
    noise_seed: int,
    soc_k: int = 1,
) -> pd.DataFrame | None:
    n = len(kin["speed_mps"])
    t_amb = np.full(n, ambient_temp_c)
    sim = simulate_trip(
        kin["speed_mps"],
        kin["accel_mps2"],
        kin["grade_rad"],
        t_amb,
        soc_0,
        dt_s,
        vehicle,
        constants,
    )
    if not sim["range_b_defined"]:
        return None
    ts = np.arange(n, dtype=float) * dt_s
    soc_mask = (np.arange(n) % max(soc_k, 1) == 0).astype(int)
    frame = {
        "timestamp": ts,
        "vehicle_id": vehicle.vehicle_id,
        "trip_id": trip_id,
        "scenario": scenario,
        "speed_mps": kin["speed_mps"],
        "accel_mps2": kin["accel_mps2"],
        "grade_rad": kin["grade_rad"],
        "elevation_m": kin["elevation_m"],
        "distance_m": kin["distance_m"],
        "stop_flag": kin["stop_flag"],
        "ambient_temp_c": t_amb,
        "mass_kg": vehicle.mass_kg,
        "drag_coefficient": vehicle.drag_coefficient,
        "frontal_area_m2": vehicle.frontal_area_m2,
        "rolling_resistance": vehicle.rolling_resistance,
        "battery_capacity_j": vehicle.e_usable_j,
        "drivetrain_efficiency": vehicle.eta_dt,
        "regen_efficiency": vehicle.eta_rg,
        "aux_power_base_w": vehicle.p_aux_base_w,
        "rotating_mass_factor": vehicle.lambda_rot,
        "p_wheel_w": sim["p_wheel_w"],
        "p_batt_w": sim["p_batt_w"],
        "soc": sim["soc"],
        "energy_remaining_j": sim["e_rem_j"],
        "range_a_m": sim["range_a_m"],
        "range_b_m": sim["range_b_m"],
        "range_valid_mask": sim["range_a_mask"],
        "soc_label_mask": soc_mask,
        "noise_seed": noise_seed,
    }
    return pd.DataFrame(frame)[list(SCHEMA)]
