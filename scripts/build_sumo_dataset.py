#!/usr/bin/env python3
"""Build MoST multi-vehicle labeled dataset (PRD §9–§10).

Uses SUMO if `sumo` is on PATH; otherwise IDM kinematics on MoST geometry
(elevation + per-vehicle vType dynamics, distinct seeds/routes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import SCHEMA, labeled_trip
from src.data.splits import assert_no_leakage, trip_level_split
from src.simulation.idm_trace import idm_trace
from src.simulation.most_net import edges_for_scenario, parse_net, polyline_of, sample_walk
from src.simulation.run_sumo import sumo_available
from src.simulation.vtypes import write_vtypes
from src.vehicles import load_all_vehicles, load_physics_constants

VEHICLES_YAML = ROOT / "configs" / "vehicles.yaml"
SIM_YAML = ROOT / "configs" / "simulation.yaml"
NET = ROOT / "data" / "raw" / "sumo_scenarios" / "MoST" / "scenario" / "in" / "most.net.xml"
OUT_DIR = ROOT / "data" / "processed"
SPLIT_PATH = ROOT / "data" / "splits" / "trip_splits.json"
REPORT = ROOT / "dataset" / "reports" / "sumo_most_dataset.json"
DOC = ROOT / "docs" / "sumo_dataset.md"


def _speed_corr(df: pd.DataFrame) -> float:
    """Max |corr| of speed traces across different vehicles (PRD §9.3)."""
    by_vid: dict[str, list[np.ndarray]] = {}
    for (vid, tid), g in df.groupby(["vehicle_id", "trip_id"], sort=False):
        by_vid.setdefault(str(vid), []).append(g["speed_mps"].to_numpy())
    vids = list(by_vid)
    mx = 0.0
    for i, a in enumerate(vids):
        for b in vids[i + 1 :]:
            for ta in by_vid[a][:8]:
                for tb in by_vid[b][:8]:
                    n = min(len(ta), len(tb))
                    c = abs(float(np.corrcoef(ta[:n], tb[:n])[0, 1]))
                    if c > mx:
                        mx = c
    return mx


def main() -> None:
    sim_cfg = yaml.safe_load(SIM_YAML.read_text(encoding="utf-8"))
    n_trips = int(sim_cfg.get("n_trips_per_vehicle_scenario", 20))
    duration = float(sim_cfg.get("trip_duration_s", 600))
    dt = float(sim_cfg.get("dt_s", 1.0))
    t_amb = float(sim_cfg.get("ambient_temp_c", 21.0))
    seed = int(sim_cfg.get("seed", 42))
    scenarios = list(sim_cfg.get("scenarios") or ["urban", "arterial", "highway"])
    rng = np.random.default_rng(seed)

    vehicles = load_all_vehicles(VEHICLES_YAML)
    constants = load_physics_constants(VEHICLES_YAML)
    write_vtypes(ROOT / "simulation" / "vehicles" / "generated_vtypes.add.xml", vehicles)

    engine = "sumo" if sumo_available() else "idm_most"
    if not NET.exists():
        raise FileNotFoundError(f"MoST net missing: {NET}")
    print(f"engine={engine} parsing {NET} ...")
    edges = parse_net(NET)

    frames = []
    dropped = 0
    for vid, vehicle in vehicles.items():
        for scenario in scenarios:
            pool = edges_for_scenario(edges, scenario)
            if not pool:
                raise RuntimeError(f"no edges for scenario {scenario}")
            for k in range(n_trips):
                trip_seed = seed + 17 * k + 101 * scenarios.index(scenario) + 1009 * list(vehicles).index(vid)
                trip_rng = np.random.default_rng(trip_seed)
                trip_id = f"{vid}__{scenario}__{k:02d}"
                ok = False
                for _try in range(8):
                    walk = sample_walk(edges, pool, trip_rng, min_length_m=duration * 8.0)
                    s_poly, xyz = polyline_of(walk, edges)
                    v_lim = float(np.median([edges[e].speed_mps for e in walk]))
                    kin = idm_trace(
                        s_poly, xyz, duration, dt, vehicle, v_lim, trip_rng, scenario
                    )
                    soc0 = float(trip_rng.uniform(0.55, 0.95))
                    part = labeled_trip(
                        kin,
                        vehicle,
                        constants,
                        trip_id=trip_id,
                        scenario=scenario,
                        dt_s=dt,
                        ambient_temp_c=t_amb,
                        soc_0=soc0,
                        noise_seed=trip_seed,
                    )
                    if part is not None:
                        frames.append(part)
                        ok = True
                        break
                if not ok:
                    dropped += 1
                    print(f"drop {trip_id} (net-negative energy)")

    df = pd.concat(frames, ignore_index=True)
    z_std = float(df["elevation_m"].std())
    if z_std <= 1.0:
        raise RuntimeError(f"elevation std {z_std:.3f} m <= 1.0 (PRD §9.2)")

    keys = df[["trip_id", "vehicle_id", "scenario"]].drop_duplicates()
    split = trip_level_split(keys, np.random.default_rng(seed + 1))
    assert_no_leakage(split)
    max_corr = _speed_corr(df)
    if max_corr >= 0.9:
        raise RuntimeError(
            f"cross-vehicle speed-trace |corr| = {max_corr:.3f} (>= 0.9); PRD §9.3"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "sumo_most_1hz.csv.gz"
    df.to_csv(out_csv, index=False, compression="gzip")
    SPLIT_PATH.write_text(json.dumps(split, indent=2), encoding="utf-8")

    summary = {
        "engine": engine,
        "sumo_available": sumo_available(),
        "rows": int(len(df)),
        "trips": int(df["trip_id"].nunique()),
        "vehicles": sorted(df["vehicle_id"].unique().tolist()),
        "scenarios": sorted(df["scenario"].unique().tolist()),
        "elevation_std_m": z_std,
        "elevation_minmax_m": [float(df["elevation_m"].min()), float(df["elevation_m"].max())],
        "max_abs_speed_corr": max_corr,
        "dropped_negative_energy_trips": dropped,
        "split_counts": {k: len(v) for k, v in split.items()},
        "schema": list(SCHEMA),
        "output": str(out_csv.relative_to(ROOT)),
        "note": "Labels from project physics E1–E14, not SUMO battery device.",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    DOC.write_text(
        "\n".join(
            [
                "# MoST multi-vehicle dataset (Tier C)",
                "",
                f"- Engine: **{engine}** (`sumo` on PATH: {sumo_available()})",
                f"- Rows: {summary['rows']:,} at dt={dt}s; trips: {summary['trips']}",
                f"- Vehicles: {', '.join(summary['vehicles'])}",
                f"- Scenarios: {', '.join(summary['scenarios'])}",
                f"- Elevation std: **{z_std:.2f} m** (PRD gate > 1.0 m)",
                f"- Max |corr| of trip speed traces: {max_corr:.3f} (gate < 0.9)",
                f"- Splits (trips): {summary['split_counts']}",
                f"- Dropped net-negative energy trips: {dropped}",
                "",
                "Network: MoST `most.net.xml` (Codeca & Härri). Physics labels: `src/physics/`.",
                "If SUMO is installed later, re-run this script; FCD+z replaces IDM speeds",
                "but elevation still comes from the same net.",
                "",
                "Outputs: `data/processed/sumo_most_1hz.csv.gz`, `data/splits/trip_splits.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
