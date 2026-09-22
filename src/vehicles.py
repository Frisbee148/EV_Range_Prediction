"""Load `configs/vehicles.yaml` and convert units once (PRD §5, §8, §22.1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

KWH_TO_J = 3.6e6
KW_TO_W = 1.0e3

REQUIRED_KEYS = (
    "mass_kg",
    "drag_coefficient",
    "frontal_area_m2",
    "rolling_resistance",
    "battery_capacity_kwh",
    "drivetrain_efficiency",
    "regen_efficiency",
    "regen_power_max_kw",
    "aux_power_kw",
    "rotating_mass_factor",
)

ALLOWED_TIERS = frozenset({"SOURCED", "DERIVED", "ASSUMED"})


@dataclass(frozen=True)
class PhysicsConstants:
    g: float
    rho: float
    t_comfort_c: float
    k_hvac_w_per_c: float
    v_min_mps: float
    v_regen_min_mps: float
    eps: float
    ema_window_s: float
    ema_init_s: float
    d_min_m: float


@dataclass(frozen=True)
class VehicleParams:
    vehicle_id: str
    display_name: str
    mass_kg: float
    drag_coefficient: float
    frontal_area_m2: float
    rolling_resistance: float
    e_usable_j: float
    eta_dt: float
    eta_rg: float
    p_regen_max_w: float
    p_aux_base_w: float
    lambda_rot: float
    max_speed_mps: float
    accel_mps2: float
    decel_mps2: float
    sigma: float
    provenance: dict[str, dict[str, Any]]


def _unwrap(node: Any, key: str) -> tuple[Any, dict[str, Any]]:
    if isinstance(node, dict) and "value" in node:
        if "tier" not in node:
            raise ValueError(f"missing provenance tier for {key}")
        tier = str(node["tier"]).upper()
        if tier not in ALLOWED_TIERS:
            raise ValueError(f"invalid tier {tier!r} for {key}")
        meta = {
            "tier": tier,
            "citation": node.get("citation", ""),
            "note": node.get("note", ""),
        }
        return node["value"], meta
    raise ValueError(f"{key} must be a mapping with value and tier")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"invalid yaml at {path}")
    return data


def load_physics_constants(path: Path | str) -> PhysicsConstants:
    raw = _load_yaml(Path(path))
    c = raw.get("constants") or {}
    return PhysicsConstants(
        g=float(c["g"]),
        rho=float(c["rho"]),
        t_comfort_c=float(c.get("t_comfort_c", 21.0)),
        k_hvac_w_per_c=float(c.get("k_hvac_w_per_c", 60.0)),
        v_min_mps=float(c.get("v_min_mps", 0.5)),
        v_regen_min_mps=float(c.get("v_regen_min_mps", 5.0)),
        eps=float(c.get("eps", 1e-6)),
        ema_window_s=float(c.get("ema_window_s", 300.0)),
        ema_init_s=float(c.get("ema_init_s", 60.0)),
        d_min_m=float(c.get("d_min_m", 500.0)),
    )


def load_vehicle(path: Path | str, vehicle_id: str) -> VehicleParams:
    raw = _load_yaml(Path(path))
    vehicles = raw.get("vehicles") or {}
    if vehicle_id not in vehicles:
        raise KeyError(f"unknown vehicle_id {vehicle_id!r}")
    block = vehicles[vehicle_id]
    provenance: dict[str, dict[str, Any]] = {}
    values: dict[str, Any] = {}
    for key in REQUIRED_KEYS:
        val, meta = _unwrap(block[key], f"{vehicle_id}.{key}")
        values[key] = float(val)
        provenance[key] = meta
    sumo = block.get("sumo") or {}
    return VehicleParams(
        vehicle_id=vehicle_id,
        display_name=str(block.get("display_name", vehicle_id)),
        mass_kg=values["mass_kg"],
        drag_coefficient=values["drag_coefficient"],
        frontal_area_m2=values["frontal_area_m2"],
        rolling_resistance=values["rolling_resistance"],
        e_usable_j=values["battery_capacity_kwh"] * KWH_TO_J,
        eta_dt=values["drivetrain_efficiency"],
        eta_rg=values["regen_efficiency"],
        p_regen_max_w=values["regen_power_max_kw"] * KW_TO_W,
        p_aux_base_w=values["aux_power_kw"] * KW_TO_W,
        lambda_rot=values["rotating_mass_factor"],
        max_speed_mps=float(sumo.get("max_speed_mps", 33.0)),
        accel_mps2=float(sumo.get("accel_mps2", 2.6)),
        decel_mps2=float(sumo.get("decel_mps2", 4.5)),
        sigma=float(sumo.get("sigma", 0.5)),
        provenance=provenance,
    )


def load_all_vehicles(path: Path | str) -> dict[str, VehicleParams]:
    raw = _load_yaml(Path(path))
    ids = list((raw.get("vehicles") or {}).keys())
    return {vid: load_vehicle(path, vid) for vid in ids}
