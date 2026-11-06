"""SUMO battery-device cross-check and XML parser (PRD §9.4).

Compares `simulate_trip` from `src/physics/` against SUMO's `device.battery`
formulation (`MSDevice_Battery`) and/or a `--battery-output` XML trace,
verifying cumulative energy agreement within 2%.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.physics.trip import simulate_trip
from src.vehicles import KWH_TO_J, PhysicsConstants, VehicleParams


def parse_sumo_battery_xml(xml_path: str | Path, vehicle_id: str | None = None) -> pd.DataFrame:
    """Parse a SUMO `--battery-output` XML file into a DataFrame."""
    tree = ET.parse(Path(xml_path))
    root = tree.getroot()
    rows: list[dict[str, Any]] = []

    for ts_elem in root.findall("timestep"):
        t_s = float(ts_elem.attrib.get("time", 0.0))
        for veh_elem in ts_elem.findall("vehicle"):
            vid = str(veh_elem.attrib.get("id", ""))
            if vehicle_id is not None and vid != vehicle_id:
                continue
            e_wh = float(veh_elem.attrib.get("energyConsumed", 0.0))
            act_wh = float(veh_elem.attrib.get("actualBatteryCapacity", 0.0))
            max_wh = float(veh_elem.attrib.get("maximumBatteryCapacity", 1.0))
            soc = act_wh / max(max_wh, 1e-9)
            rows.append(
                {
                    "time_s": t_s,
                    "vehicle_id": vid,
                    "energy_consumed_wh": e_wh,
                    "actual_battery_capacity_wh": act_wh,
                    "max_battery_capacity_wh": max_wh,
                    "soc": soc,
                    "speed_mps": float(veh_elem.attrib.get("speed", 0.0)),
                    "accel_mps2": float(veh_elem.attrib.get("acceleration", 0.0)),
                    "slope_deg": float(veh_elem.attrib.get("slope", 0.0)),
                }
            )
    return pd.DataFrame(rows)


def sumo_battery_device_power(
    v_mps: np.ndarray,
    a_mps2: np.ndarray,
    theta_rad: np.ndarray,
    t_amb_c: np.ndarray,
    dt_s: float,
    vehicle: VehicleParams,
    constants: PhysicsConstants,
) -> np.ndarray:
    """Evaluate SUMO's `device.battery` (`EnergyParams::compute`) step power in Watts.

    Maps `VehicleParams` 1-to-1 to SUMO vType battery parameters:
    - `vehicleMass` + `internalMomentOfInertia` (`m * lambda_rot`)
    - `frontSurfaceArea` (`A`), `airDragCoefficient` (`C_d`)
    - `rollDragCoefficient` (`C_rr`)
    - `propulsionEfficiency` (`eta_dt`), `recuperationEfficiency` (`eta_rg`)
    - `constantPowerIntake` (`P_aux` with HVAC offset)
    """
    v = np.maximum(np.asarray(v_mps, dtype=float), 0.0)
    a = np.asarray(a_mps2, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    t_amb = np.asarray(t_amb_c, dtype=float)

    # SUMO EnergyParams step work ΔE over dt_s (converted to average step power in W)
    ds = v * dt_s
    e_air_j = 0.5 * constants.rho * vehicle.drag_coefficient * vehicle.frontal_area_m2 * (v ** 2) * ds
    e_roll_j = vehicle.rolling_resistance * vehicle.mass_kg * constants.g * np.cos(theta) * ds
    e_pot_j = vehicle.mass_kg * constants.g * np.sin(theta) * ds
    e_kin_j = vehicle.mass_kg * vehicle.lambda_rot * a * ds

    p_mech_w = (e_air_j + e_roll_j + e_pot_j + e_kin_j) / max(dt_s, 1e-9)

    p_prop_w = p_mech_w / vehicle.eta_dt
    regen_active = v >= constants.v_regen_min_mps
    p_regen_raw_w = np.where(regen_active, p_mech_w * vehicle.eta_rg, 0.0)
    p_regen_w = -np.minimum(np.abs(p_regen_raw_w), vehicle.p_regen_max_w)

    p_aux_w = vehicle.p_aux_base_w + constants.k_hvac_w_per_c * np.abs(
        t_amb - constants.t_comfort_c
    )
    return np.where(p_mech_w >= 0.0, p_prop_w, p_regen_w) + p_aux_w


def compare_to_sumo_battery(
    v_mps: np.ndarray,
    a_mps2: np.ndarray,
    theta_rad: np.ndarray,
    t_amb_c: np.ndarray,
    soc_0: float,
    dt_s: float,
    vehicle: VehicleParams,
    constants: PhysicsConstants,
    *,
    sumo_battery_xml: str | Path | None = None,
    tolerance_rel: float = 0.02,
) -> dict[str, Any]:
    """Compare `simulate_trip` against SUMO `device.battery` and verify <= 2% energy difference (PRD §9.4)."""
    sim = simulate_trip(
        v_mps=v_mps,
        a_mps2=a_mps2,
        theta_rad=theta_rad,
        t_amb_c=t_amb_c,
        soc_0=soc_0,
        dt_s=dt_s,
        vehicle=vehicle,
        constants=constants,
    )
    e_phys_kwh = float(np.sum(sim["p_batt_w"][:-1] * dt_s) / KWH_TO_J)

    if sumo_battery_xml is not None:
        df_xml = parse_sumo_battery_xml(sumo_battery_xml)
        e_sumo_kwh = float(df_xml["energy_consumed_wh"].sum() / 1000.0)
        p_sumo_w = df_xml["energy_consumed_wh"].to_numpy(dtype=float) * 3600.0 / max(dt_s, 1e-9)
    else:
        p_sumo_w = sumo_battery_device_power(
            v_mps=v_mps,
            a_mps2=a_mps2,
            theta_rad=theta_rad,
            t_amb_c=t_amb_c,
            dt_s=dt_s,
            vehicle=vehicle,
            constants=constants,
        )
        e_sumo_kwh = float(np.sum(p_sumo_w[:-1] * dt_s) / KWH_TO_J)

    rel_diff = abs(e_phys_kwh - e_sumo_kwh) / max(abs(e_sumo_kwh), 1e-9)
    max_power_diff_w = float(np.max(np.abs(sim["p_batt_w"] - p_sumo_w)))
    passes = bool(rel_diff <= tolerance_rel)

    return {
        "vehicle_id": vehicle.vehicle_id,
        "physics_energy_kwh": e_phys_kwh,
        "sumo_battery_energy_kwh": e_sumo_kwh,
        "relative_energy_diff": float(rel_diff),
        "max_step_power_diff_w": max_power_diff_w,
        "tolerance_rel": float(tolerance_rel),
        "passes_2pct_gate": passes,
    }
