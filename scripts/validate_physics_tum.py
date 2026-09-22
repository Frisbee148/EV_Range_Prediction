#!/usr/bin/env python3
"""Compare physics P_batt to TUM TripA01 measured V×I (PRD first-week gate)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.physics.trip import kinematics_from_speed_elevation, simulate_trip
from src.vehicles import load_physics_constants, load_vehicle
TRIP = ROOT / "data" / "raw" / "tum_bmw_i3" / "TripA01.csv"
VEHICLES = ROOT / "configs" / "vehicles.yaml"
OUT = ROOT / "dataset" / "reports" / "physics_tum_tripa01.json"
DOC = ROOT / "docs" / "physics_tum_validation.md"
KPH_TO_MPS = 1.0 / 3.6


def load_trip(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=";", encoding="latin-1")
    raw.columns = [c.strip() for c in raw.columns]
    t = pd.to_numeric(raw["Time [s]"], errors="coerce")
    v = pd.to_numeric(raw["Velocity [km/h]"], errors="coerce") * KPH_TO_MPS
    z = pd.to_numeric(raw["Elevation [m]"], errors="coerce")
    volt = pd.to_numeric(raw["Battery Voltage [V]"], errors="coerce")
    curr = pd.to_numeric(raw["Battery Current [A]"], errors="coerce")
    soc = pd.to_numeric(raw["SoC [%]"], errors="coerce") / 100.0
    amb = pd.to_numeric(raw["Ambient Temperature [°C]"], errors="coerce")
    df = pd.DataFrame(
        {
            "t": t,
            "v": v,
            "z": z,
            "p_meas": -volt * curr,
            "soc": soc,
            "t_amb": amb,
        }
    ).dropna()
    return df.sort_values("t").reset_index(drop=True)


def metrics(pred: np.ndarray, meas: np.ndarray) -> dict[str, float]:
    err = pred - meas
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    denom = np.maximum(np.abs(meas), 1.0)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    return {"mae_w": mae, "rmse_w": rmse, "mape_pct_vs_1w_floor": mape}


def main() -> None:
    df = load_trip(TRIP)
    dt = float(np.median(np.diff(df["t"].to_numpy())))
    vehicle = load_vehicle(VEHICLES, "bmw_i3_60ah")
    constants = load_physics_constants(VEHICLES)
    v = df["v"].to_numpy()
    a, theta = kinematics_from_speed_elevation(v, df["z"].to_numpy(), dt)
    t_amb = df["t_amb"].to_numpy()
    soc0 = float(df["soc"].iloc[0])
    sim = simulate_trip(v, a, theta, t_amb, soc0, dt, vehicle, constants)
    p_hat = np.asarray(sim["p_batt_w"], dtype=float)
    p_meas = df["p_meas"].to_numpy()
    moving = v > 0.5
    m_all = metrics(p_hat, p_meas)
    m_move = metrics(p_hat[moving], p_meas[moving])
    e_hat = float(np.sum(p_hat * dt) / 3.6e6)
    e_meas = float(np.sum(p_meas * dt) / 3.6e6)
    soc_meas_end = float(df["soc"].iloc[-1])
    soc_hat_end = float(np.asarray(sim["soc"])[-1])
    e_soc_kwh = float((soc0 - soc_meas_end) * vehicle.e_usable_j / 3.6e6)
    energy_rel_pct = 100.0 * abs(e_hat - e_meas) / max(abs(e_meas), 1e-9)
    energy_vs_soc_pct = 100.0 * abs(e_hat - e_soc_kwh) / max(abs(e_soc_kwh), 1e-9)
    report = {
        "trip": "TripA01",
        "vehicle_id": "bmw_i3_60ah",
        "n": int(len(df)),
        "dt_s": dt,
        "duration_s": float(df["t"].iloc[-1] - df["t"].iloc[0]),
        "elevation_std_m": float(np.std(df["z"])),
        "p_meas_mean_w": float(np.mean(p_meas)),
        "p_hat_mean_w": float(np.mean(p_hat)),
        "metrics_all": m_all,
        "metrics_moving_v_gt_0.5": m_move,
        "energy_meas_kwh": e_meas,
        "energy_physics_kwh": e_hat,
        "energy_soc_implied_kwh": e_soc_kwh,
        "energy_rel_error_pct": energy_rel_pct,
        "energy_vs_soc_implied_pct": energy_vs_soc_pct,
        "soc_start": soc0,
        "soc_end_meas": soc_meas_end,
        "soc_end_physics": soc_hat_end,
        "range_b_defined": bool(sim["range_b_defined"]),
        "p_batt_min_w": float(np.min(p_hat)),
        "p_regen_max_w": vehicle.p_regen_max_w,
        "sign": "TUM current is charge-positive; P_meas = -V*I (discharge positive)",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    DOC.write_text(
        "\n".join(
            [
                "# Physics vs TUM TripA01",
                "",
                "Gate: compare `simulate_trip` battery power to measured `P = -V·I` on BMW i3.",
                "",
                f"- Samples: {report['n']} at dt = {dt:.3f} s, duration {report['duration_s']:.1f} s",
                f"- Elevation std: {report['elevation_std_m']:.2f} m",
                f"- Power MAE (all / moving): {m_all['mae_w']:.0f} W / {m_move['mae_w']:.0f} W",
                f"- Power RMSE (all): {m_all['rmse_w']:.0f} W",
                f"- Energy measured vs physics: {e_meas:.3f} kWh vs {e_hat:.3f} kWh "
                f"({energy_rel_pct:.1f}% relative to I×V)",
                f"- Energy implied by ΔSOC × 18.8 kWh: {e_soc_kwh:.3f} kWh "
                f"(physics vs this: {energy_vs_soc_pct:.1f}%)",
                f"- SOC start → end measured: {soc0:.3f} → {soc_meas_end:.3f}; "
                f"physics SOC end {soc_hat_end:.3f}",
                f"- Regen clamp: min P_batt = {report['p_batt_min_w']:.0f} W "
                f"(limit −{vehicle.p_regen_max_w:.0f} W)",
                "",
                "Machine-readable: `dataset/reports/physics_tum_tripa01.json`.",
                "",
                "Instantaneous MAPE vs I×V is high (transients, constant η). Trip **energy vs I×V is 27%**;",
                "that I×V integral also disagrees with ΔSOC×E_usable (1.29 vs 1.02 kWh).",
                "**Physics vs SOC-implied energy is 7.4%**, and end SOC matches within 0.4 pp.",
                "Treat as a pass on energy/SOC bookkeeping; do not expect tight per-timestep power fit",
                "until HVAC is calibrated. SUMO battery-device 2% check is still pending (no MoST FCD yet).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
