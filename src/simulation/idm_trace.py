"""Single-vehicle IDM traces on a MoST polyline (PRD §9.3 / §9.5)."""

from __future__ import annotations

import numpy as np

from src.vehicles import VehicleParams


def idm_trace(
    s_poly: np.ndarray,
    xyz: np.ndarray,
    duration_s: float,
    dt_s: float,
    vehicle: VehicleParams,
    edge_speed_mps: float,
    rng: np.random.Generator,
    scenario: str,
) -> dict[str, np.ndarray]:
    n = int(round(duration_s / dt_s))
    v0 = min(vehicle.max_speed_mps, max(edge_speed_mps, 5.0))
    a_max = vehicle.accel_mps2
    b = vehicle.decel_mps2
    v = np.zeros(n, dtype=float)
    s = np.zeros(n, dtype=float)
    speed = float(rng.uniform(0.0, 0.4 * v0))
    pos = 0.0
    s_max = float(s_poly[-1])
    stop_until = -1.0
    next_stop = _first_stop(scenario, rng)
    block = max(8, int(rng.integers(12, 40)))
    lo, hi = _traffic_span(scenario)
    n_blocks = n // block + 2
    factors = rng.uniform(lo, hi, size=n_blocks)
    vcap = np.repeat(factors, block)[:n] * v0
    for t in range(n):
        time_s = t * dt_s
        if time_s >= next_stop and stop_until < 0:
            stop_until = time_s + float(rng.uniform(*_dwell(scenario)))
            next_stop = time_s + float(rng.uniform(*_gap(scenario)))
        if 0 <= stop_until and time_s < stop_until:
            acc = -b
        else:
            stop_until = -1.0
            cap = max(float(vcap[t]), 0.3)
            ratio = speed / cap
            acc = a_max * (1.0 - ratio**4)
        acc += rng.normal(0.0, vehicle.sigma * a_max * 0.15)
        acc = float(np.clip(acc, -b, a_max))
        speed = float(np.clip(speed + acc * dt_s, 0.0, v0))
        pos = pos + speed * dt_s
        if pos >= s_max:
            pos = pos % s_max
        v[t] = speed
        s[t] = pos
    z = np.interp(s, s_poly, xyz[:, 2])
    a = np.gradient(v, dt_s)
    dx = np.maximum(v * dt_s, 1e-9)
    dz = np.diff(z, prepend=z[0])
    theta = np.arctan2(dz, dx)
    return {
        "speed_mps": v,
        "accel_mps2": a,
        "elevation_m": z,
        "grade_rad": theta,
        "distance_m": np.cumsum(v * dt_s),
        "stop_flag": (v < 0.1).astype(int),
    }


def _traffic_span(scenario: str) -> tuple[float, float]:
    if scenario == "urban":
        return (0.12, 1.0)
    if scenario == "arterial":
        return (0.35, 1.0)
    return (0.45, 1.0)


def _first_stop(scenario: str, rng: np.random.Generator) -> float:
    lo, hi = _gap(scenario)
    return float(rng.uniform(lo * 0.3, hi))


def _gap(scenario: str) -> tuple[float, float]:
    if scenario == "urban":
        return (40.0, 90.0)
    if scenario == "arterial":
        return (90.0, 180.0)
    return (250.0, 450.0)


def _dwell(scenario: str) -> tuple[float, float]:
    if scenario == "urban":
        return (8.0, 18.0)
    if scenario == "arterial":
        return (4.0, 10.0)
    return (0.0, 3.0)
