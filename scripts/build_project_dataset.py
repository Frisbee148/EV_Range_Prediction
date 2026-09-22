#!/usr/bin/env python3
"""
Build a unified, cleaned project dataset from:
  - Argonne D3 (Tier B)
  - TUM BMW i3 (Tier A)
aligned toward PRD §10 schema fields that real telemetry can support.

Outputs under dataset/:
  processed/d3_unified.parquet|csv.gz
  processed/tum_unified.parquet|csv.gz
  processed/project_unified.parquet|csv.gz
  reports/attribute_provenance.md
  reports/build_summary.json
"""
from __future__ import annotations

import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_D3 = ROOT / "data" / "raw" / "argonne_d3"
RAW_TUM = ROOT / "data" / "raw" / "tum_bmw_i3"
OUT = ROOT / "dataset"
PROC = OUT / "processed"
REP = OUT / "reports"

MPH_TO_MPS = 0.44704
KPH_TO_MPS = 1.0 / 3.6
MI_TO_M = 1609.344

# Classic D3 column aliases (older txt packs)
CLASSIC_MAP = {
    "time_s": ["Time[sec]", "Time[s]", "Time"],
    "speed_mph": ["Dyno_Speed[mph]", "Dyno_Spd[mph]"],
    "tractive_n": ["Dyno_Tractive_Effort[N]", "Dyno_TractiveForce[N]"],
    "temp_c": ["Test_Cell_Temp[C]", "Cell_Temp[C]"],
    "soc_pct": ["HV_Battery_SOC[%]", "HV_Battery_SOC"],
    "current_a": ["HV_Battery_Current[A]"],
    "voltage_v": ["HV_Battery_Voltage[V]"],
    "pedal_pct": ["Accelerator_Pedal_Position[%]"],
}

# Extended Leaf/Bolt preferred columns (Hioki / CAN)
EXTENDED_PREF = {
    "leaf": {
        "time_s": ["Time[s]_RawFacilities", "Time[s]", "DAQ_Time[s]"],
        "speed_mph": ["Dyno_Spd[mph]"],
        "tractive_n": ["Dyno_TractiveForce[N]"],
        "temp_c": ["Cell_Temp[C]", "Veh_ambient_temp_CAN5__C"],
        "soc_pct": ["HVBatt_SOC_CAN5__per", "Batt_SOC_CAN2__per", "HVBatt_SOC_HEV__per"],
        "current_a": ["HVBatt_current_wide_CAN5__A", "HVBatt_current_HEV__A", "HVBatt_current_BECM__A"],
        "voltage_v": ["HVBatt_voltage_CAN5__V", "HVBatt_voltage_HEV__V", "HVBatt_volt_total_BECM__V"],
        "grade_pct": ["Grade_Output_DriveTrace[pct]"],
        "distance_mi": ["Distance[mi]"],
    },
    "bolt": {
        "time_s": ["DAQ_Time[s]", "Time[s]_RawFacilities", "Time[s]"],
        "speed_mph": ["Dyno_Spd[mph]"],
        "tractive_n": ["Dyno_TractiveForce[N]"],
        "temp_c": ["Cell_Temp[C]", "Veh_ambient_air_temp_HPCM2__C"],
        "soc_pct": ["HVBatt_SOC_CAN4__per", "HVBatt_pack_SOC_HPCM2__per", "HVBatt_SOC_HPCM__per"],
        "current_a": ["HVBatt_Curr_Hioki_I1__A", "HVBatt_pack_current_HPCM2__A", "HVBatt_current_high_CAN__A"],
        "voltage_v": ["HVBatt_Volt_Hioki_U1__V", "HVBatt_pack_voltage_HPCM2__V", "HVBatt_voltage_CAN__V"],
        "grade_pct": ["Grade_Output_DriveTrace[pct]"],
        "distance_mi": ["Distance[mi]"],
    },
}

VEHICLE_META = {
    "bmw_i3_2014": {"display": "2014 BMW i3 BEV", "source": "argonne_d3", "dir": "2014 BMW i3"},
    "kia_soul_ev_2015": {"display": "2015 Kia Soul EV", "source": "argonne_d3", "dir": "2015 Kia Soul EV"},
    "ford_focus_ev_2013": {"display": "2013 Ford Focus Electric", "source": "argonne_d3", "dir": "2013 Ford Focus Electric"},
    "vw_egolf_2015": {"display": "2015 Volkswagen e-Golf", "source": "argonne_d3", "dir": "2015 Volkswagen e-Golf"},
    "mb_bclass_ev_2015": {"display": "2015 Mercedes-Benz B-Class Electric", "source": "argonne_d3", "dir": "2015 Mercedes Benz B-Class Electric"},
    "nissan_leaf_eplus_2019": {"display": "2019 Nissan Leaf ePlus", "source": "argonne_d3", "dir": "2019 Nissan Leaf E Plus"},
    "chevy_bolt_2020": {"display": "2020 Chevrolet Bolt", "source": "argonne_d3", "dir": "2020 Chevrolet Bolt"},
    "hyundai_ioniq5_2023": {"display": "2023 Hyundai Ioniq 5", "source": "argonne_d3", "dir": "2023 Hyundai Ioniq 5"},
    "bmw_i3_tum": {"display": "BMW i3 60Ah (TUM real trips)", "source": "tum", "dir": None},
}


def pick_col(columns, candidates):
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def read_header_cols(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        line = f.readline()
    return [c.strip() for c in line.rstrip("\n\r").split("\t")]


def read_d3_txt(path: Path, wanted_candidates: list[list[str]] | None = None) -> pd.DataFrame:
    """Fast tab read; optionally load only columns matching candidate lists."""
    header = read_header_cols(path)
    usecols = None
    if wanted_candidates is not None:
        usecols = []
        for cands in wanted_candidates:
            col = pick_col(header, cands)
            if col and col not in usecols:
                usecols.append(col)
        if not usecols:
            raise ValueError(f"no matching columns in {path}")
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        engine="c",
        low_memory=False,
    )
    df.columns = [str(c).strip() for c in df.columns]
    return df


def finalize_trip(df: pd.DataFrame, vehicle_id: str, trip_id: str, source: str) -> pd.DataFrame:
    """Map raw extracted columns to PRD-aligned schema + cleaning."""
    out = pd.DataFrame()
    out["timestamp"] = pd.to_numeric(df["time_s"], errors="coerce")
    out["vehicle_id"] = vehicle_id
    out["trip_id"] = trip_id
    out["scenario"] = "dyno" if source == "argonne_d3" else "real_drive"
    out["source"] = source

    # Speed
    if "speed_mps" in df.columns:
        out["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce")
    elif "speed_mph" in df.columns:
        out["speed_mps"] = pd.to_numeric(df["speed_mph"], errors="coerce") * MPH_TO_MPS
    elif "speed_kph" in df.columns:
        out["speed_mps"] = pd.to_numeric(df["speed_kph"], errors="coerce") * KPH_TO_MPS
    else:
        out["speed_mps"] = np.nan

    # Acceleration (DERIVED)
    out = out.sort_values("timestamp").reset_index(drop=True)
    dt = out["timestamp"].diff()
    dv = out["speed_mps"].diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        accel = dv / dt
    accel[(dt <= 0) | ~np.isfinite(accel)] = np.nan
    out["accel_mps2"] = accel.clip(-10, 10)

    # Grade / elevation
    if "grade_pct" in df.columns:
        grade_pct = pd.to_numeric(df["grade_pct"], errors="coerce").fillna(0.0)
        out["grade_rad"] = np.arctan(grade_pct / 100.0)  # DERIVED from measured grade %
        out["grade_pct_raw"] = grade_pct
    else:
        out["grade_rad"] = 0.0  # D3 dyno default / TUM uses elevation instead
        out["grade_pct_raw"] = 0.0

    if "elevation_m" in df.columns:
        out["elevation_m"] = pd.to_numeric(df["elevation_m"], errors="coerce")
    else:
        out["elevation_m"] = 0.0  # D3: flat dyno

    # Distance
    if "distance_m" in df.columns:
        out["distance_m"] = pd.to_numeric(df["distance_m"], errors="coerce")
    elif "distance_mi" in df.columns:
        out["distance_m"] = pd.to_numeric(df["distance_mi"], errors="coerce") * MI_TO_M
    else:
        # integrate speed (DERIVED)
        dt_f = out["timestamp"].diff().fillna(0.0).clip(lower=0)
        out["distance_m"] = (out["speed_mps"].fillna(0) * dt_f).cumsum()

    out["stop_flag"] = (out["speed_mps"].fillna(0) < 0.1).astype(int)

    if "temp_c" in df.columns:
        out["ambient_temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    else:
        out["ambient_temp_c"] = np.nan

    # Battery measurements
    for col in ("voltage_v", "current_a", "soc_pct", "tractive_n"):
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = np.nan

    # p_batt: positive = discharge (PRD convention)
    # Classic D3 / many ANL files: positive current often = discharge.
    # TUM: charge-positive (idle current negative) → use current_sign=-1
    sign = float(df["current_sign"].iloc[0]) if "current_sign" in df.columns else 1.0
    out["p_batt_w"] = out["voltage_v"] * out["current_a"] * sign  # DERIVED from V,I
    out["soc"] = out["soc_pct"] / 100.0

    # Dyno wheel-ish power from tractive effort * speed (DERIVED)
    out["p_wheel_w"] = out["tractive_n"] * out["speed_mps"]

    # Cleaning
    out = out.replace([np.inf, -np.inf], np.nan)
    # Keep rows with valid time and speed
    out = out.dropna(subset=["timestamp", "speed_mps"])
    # Drop pre-test negative times (classic D3 pads ~10 s before t=0)
    out = out[out["timestamp"] >= 0].copy()
    # Drop exact duplicate timestamps within a trip
    out = out.drop_duplicates(subset=["timestamp"], keep="first")
    # Drop rows where ALL core battery signals missing (except speed-only Ioniq cases keep if speed ok)
    # Require at least speed + (p_batt or soc)
    has_batt = out["p_batt_w"].notna() | out["soc"].notna()
    out = out[has_batt | (out["source"] == "argonne_d3")].copy()

    # Clip SOC to [0,1] if present
    out.loc[out["soc"].notna(), "soc"] = out.loc[out["soc"].notna(), "soc"].clip(0, 1)

    # Resample-ish: if dt mostly 0.1s, keep 10 Hz; also provide 1 Hz downsample marker
    out["dt_s"] = out["timestamp"].diff().fillna(0.1)

    # Static placeholders (filled later from EPA / config — not invented here)
    for c in [
        "mass_kg",
        "drag_coefficient",
        "frontal_area_m2",
        "rolling_resistance",
        "battery_capacity_j",
        "drivetrain_efficiency",
        "regen_efficiency",
        "aux_power_base_w",
        "rotating_mass_factor",
        "energy_remaining_j",
        "range_a_m",
        "range_b_m",
    ]:
        out[c] = np.nan

    out["range_valid_mask"] = 0
    out["soc_label_mask"] = out["soc"].notna().astype(int)
    out["noise_seed"] = -1

    return out.reset_index(drop=True)


def _map_from_raw(raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    mapped = {}
    for key, cands in mapping.items():
        col = pick_col(raw.columns, cands)
        if col:
            mapped[key] = raw[col]
    return pd.DataFrame(mapped)


def process_classic_vehicle(vehicle_id: str, folder: Path) -> list[pd.DataFrame]:
    trips = []
    files = sorted(folder.rglob("*Test Data*.txt"))
    files = [f for f in files if "Charge" not in f.name]
    for f in files:
        try:
            raw = read_d3_txt(f, wanted_candidates=list(CLASSIC_MAP.values()))
        except Exception as e:
            warnings.warn(f"skip {f}: {e}")
            continue
        df = _map_from_raw(raw, CLASSIC_MAP)
        if "time_s" not in df.columns or "speed_mph" not in df.columns:
            continue
        df["current_sign"] = 1.0
        trip_id = f"{vehicle_id}__{f.stem.replace(' ', '_')}"
        trips.append(finalize_trip(df, vehicle_id, trip_id, "argonne_d3"))
        print(f"  + {trip_id} rows={len(trips[-1])}", flush=True)
    return trips


def process_extended_vehicle(vehicle_id: str, folder: Path, kind: str) -> list[pd.DataFrame]:
    trips = []
    prefs = EXTENDED_PREF[kind]
    files = sorted(folder.rglob("*Test Data*.txt"))
    for f in files:
        try:
            raw = read_d3_txt(f, wanted_candidates=list(prefs.values()))
        except Exception as e:
            warnings.warn(f"skip {f}: {e}")
            continue
        df = _map_from_raw(raw, prefs)
        if "time_s" not in df.columns or "speed_mph" not in df.columns:
            continue
        # Hioki / CAN pack current: treat positive as discharge for ANL (verify later)
        df["current_sign"] = 1.0
        trip_id = f"{vehicle_id}__{f.stem.replace(' ', '_')}"
        trips.append(finalize_trip(df, vehicle_id, trip_id, "argonne_d3"))
        print(f"  + {trip_id} rows={len(trips[-1])}", flush=True)
    return trips


def process_ioniq_tdms(vehicle_id: str, folder: Path) -> list[pd.DataFrame]:
    try:
        from nptdms import TdmsFile
    except ImportError:
        warnings.warn("nptdms not installed; skipping Ioniq TDMS")
        return []
    trips = []
    for f in sorted(folder.rglob("*.tdms")):
        try:
            td = TdmsFile.read(str(f))
        except Exception as e:
            warnings.warn(f"skip tdms {f}: {e}")
            continue
        g = td.groups()[0]
        ch = {c.name: c[:] for c in g.channels()}
        n = len(ch.get("Time", ch.get("DAQ_Time", [])))
        if n == 0:
            continue
        df = pd.DataFrame(
            {
                "time_s": ch.get("Time", ch.get("DAQ_Time")),
                "speed_mph": ch.get("Dyno_Spd"),
                "tractive_n": ch.get("Dyno_TractiveForce"),
                "temp_c": ch.get("Cell_Temp"),
                "grade_pct": ch.get("Grade_Output_DriveTrace", np.zeros(n)),
                "distance_mi": ch.get("Distance"),
                # Hioki-like: Udc1 ~ pack V, Idc1 ~ pack A, P1 ~ W
                "voltage_v": ch.get("Udc1"),
                "current_a": ch.get("Idc1"),
            }
        )
        df["soc_pct"] = np.nan  # not present in Analysis.tdms
        df["current_sign"] = 1.0
        trip_id = f"{vehicle_id}__{f.stem.replace(' ', '_')}"
        trips.append(finalize_trip(df, vehicle_id, trip_id, "argonne_d3"))
    return trips


def process_tum() -> list[pd.DataFrame]:
    trips = []
    files = sorted(RAW_TUM.glob("Trip*.csv"))
    for f in files:
        try:
            raw = pd.read_csv(f, sep=";", encoding="latin-1")
        except Exception as e:
            warnings.warn(f"skip tum {f}: {e}")
            continue
        raw.columns = [c.strip() for c in raw.columns]

        def col(*names):
            for n in names:
                if n in raw.columns:
                    return raw[n]
            # fuzzy
            for c in raw.columns:
                for n in names:
                    if n.lower() in c.lower():
                        return raw[c]
            return None

        t = col("Time [s]")
        v = col("Velocity [km/h]")
        elev = col("Elevation [m]")
        amb = col("Ambient Temperature [°C]", "Ambient Temperature [C]")
        volt = col("Battery Voltage [V]")
        curr = col("Battery Current [A]")
        soc = col("SoC [%]")
        if t is None or v is None:
            continue
        df = pd.DataFrame(
            {
                "time_s": pd.to_numeric(t, errors="coerce"),
                "speed_kph": pd.to_numeric(v, errors="coerce"),
                "elevation_m": pd.to_numeric(elev, errors="coerce") if elev is not None else np.nan,
                "temp_c": pd.to_numeric(amb, errors="coerce") if amb is not None else np.nan,
                "voltage_v": pd.to_numeric(volt, errors="coerce") if volt is not None else np.nan,
                "current_a": pd.to_numeric(curr, errors="coerce") if curr is not None else np.nan,
                "soc_pct": pd.to_numeric(soc, errors="coerce") if soc is not None else np.nan,
            }
        )
        # Grade from elevation (DERIVED)
        df = df.sort_values("time_s")
        dz = df["elevation_m"].diff()
        # approximate horizontal distance from speed
        dt = df["time_s"].diff()
        ds = (pd.to_numeric(df["speed_kph"], errors="coerce") * KPH_TO_MPS) * dt
        with np.errstate(divide="ignore", invalid="ignore"):
            grade = np.arctan2(dz, ds.replace(0, np.nan))
        df["grade_pct"] = np.tan(grade) * 100.0
        df["current_sign"] = -1.0  # TUM: charge-positive → flip to discharge-positive
        trip_id = f"bmw_i3_tum__{f.stem}"
        trips.append(finalize_trip(df, "bmw_i3_tum", trip_id, "tum"))
    return trips


def downsample_1hz(df: pd.DataFrame) -> pd.DataFrame:
    """Keep ~1 Hz samples for sequence model friendliness (PRD dt default 1.0)."""
    if df.empty:
        return df
    parts = []
    for (vid, tid), g in df.groupby(["vehicle_id", "trip_id"], sort=False):
        g = g.sort_values("timestamp")
        # bucket by floor(timestamp)
        g = g.copy()
        g["_bin"] = np.floor(g["timestamp"].to_numpy()).astype(int)
        parts.append(g.groupby("_bin", as_index=False).first().drop(columns=["_bin"]))
    return pd.concat(parts, ignore_index=True) if parts else df


def write_outputs(df: pd.DataFrame, stem: str) -> dict:
    PROC.mkdir(parents=True, exist_ok=True)
    stats = {
        "rows": int(len(df)),
        "trips": int(df["trip_id"].nunique()) if len(df) else 0,
        "vehicles": sorted(df["vehicle_id"].unique().tolist()) if len(df) else [],
        "missing_frac": {},
    }
    if len(df):
        for c in df.columns:
            stats["missing_frac"][c] = float(df[c].isna().mean())
        pq = PROC / f"{stem}.parquet"
        csv = PROC / f"{stem}.csv.gz"
        try:
            df.to_parquet(pq, index=False)
            stats["parquet"] = str(pq.relative_to(ROOT))
        except Exception:
            # parquet optional
            pass
        df.to_csv(csv, index=False, compression="gzip")
        stats["csv"] = str(csv.relative_to(ROOT))
        # per-vehicle split
        vdir = PROC / stem
        vdir.mkdir(exist_ok=True)
        for vid, g in df.groupby("vehicle_id"):
            g.to_csv(vdir / f"{vid}.csv.gz", index=False, compression="gzip")
    return stats


def build_attribute_report(d3: pd.DataFrame, tum: pd.DataFrame, unified: pd.DataFrame) -> str:
    """Document DATA vs DERIVED for every project attribute."""
    rows = [
        # identifiers
        ("timestamp", "DATA", "Trip-relative time from D3/TUM clocks", "Both"),
        ("vehicle_id", "DATA", "Assigned from source package / trip file", "Both"),
        ("trip_id", "DATA", "Assigned from source filename / test id", "Both"),
        ("scenario", "DATA", "Tag: dyno (D3) or real_drive (TUM)", "Both"),
        ("source", "DATA", "argonne_d3 | tum", "Both"),
        # dynamic
        ("speed_mps", "DERIVED", "From Dyno_Speed[mph] or Velocity[km/h] via unit conversion", "Both"),
        ("accel_mps2", "DERIVED", "Finite difference of speed_mps / dt; clipped ±10", "Both"),
        ("grade_rad", "DERIVED", "D3: arctan(grade_pct/100) or 0; TUM: from Δelevation/Δs", "Both"),
        ("grade_pct_raw", "DATA", "D3 Grade_Output_DriveTrace when present; else 0", "D3 primarily"),
        ("elevation_m", "DATA", "TUM Elevation[m]; D3 set to 0 (flat dyno)", "TUM data / D3 constant"),
        ("distance_m", "MIXED", "DATA if Distance[mi] present; else DERIVED ∫v dt", "Both"),
        ("stop_flag", "DERIVED", "1 if speed_mps < 0.1", "Both"),
        ("ambient_temp_c", "DATA", "Test cell / ambient temperature", "Both"),
        ("voltage_v", "DATA", "HV battery voltage (instrumented)", "Both"),
        ("current_a", "DATA", "HV battery current (instrumented)", "Both"),
        ("soc_pct", "DATA", "Reported SOC % (missing on Ioniq TDMS Analysis)", "Most D3 + TUM"),
        ("soc", "DERIVED", "soc_pct / 100", "When SOC present"),
        ("tractive_n", "DATA", "Dyno tractive effort [N] (D3 only)", "D3"),
        ("p_batt_w", "DERIVED", "voltage_v × current_a × sign (+ = discharge)", "Both"),
        ("p_wheel_w", "DERIVED", "tractive_n × speed_mps (D3); NaN for TUM", "D3"),
        ("dt_s", "DERIVED", "timestamp.diff()", "Both"),
        # static (not filled from telemetry — from EPA/config later)
        ("mass_kg", "CONFIG/DERIVED", "Not in telemetry; fill from EPA ETW / vehicles.yaml", "Pending"),
        ("drag_coefficient", "CONFIG/DERIVED", "From EPA Cd·A split / literature", "Pending"),
        ("frontal_area_m2", "CONFIG/DERIVED", "From EPA Cd·A split / literature", "Pending"),
        ("rolling_resistance", "CONFIG/DERIVED", "From EPA target-A", "Pending"),
        ("battery_capacity_j", "CONFIG", "Usable pack energy from datasheet/EPA", "Pending"),
        ("drivetrain_efficiency", "CONFIG/ASSUMED", "Not measured in D3/TUM packs used here", "Pending"),
        ("regen_efficiency", "CONFIG/ASSUMED", "Not directly provided", "Pending"),
        ("aux_power_base_w", "CONFIG/ASSUMED", "Calibrate later from idle V×I", "Pending"),
        ("rotating_mass_factor", "CONFIG/ASSUMED", "PRD default 1.04", "Pending"),
        # labels that SUMO physics would provide — NOT invented from dyno alone
        ("energy_remaining_j", "NOT YET", "Needs battery_capacity_j × soc (DERIVED once capacity set)", "Pending"),
        ("range_a_m", "NOT YET", "Needs consumption model E13", "Pending"),
        ("range_b_m", "NOT YET", "PRIMARY label in PRD; needs cycle replay E14", "Pending"),
        ("range_valid_mask", "DERIVED", "0 until range labels exist", "Both"),
        ("soc_label_mask", "DERIVED", "1 where soc present", "Both"),
        ("noise_seed", "DERIVED", "Placeholder -1 (no §11.3 noise applied yet)", "Both"),
    ]

    n_data = sum(1 for r in rows if r[1] == "DATA")
    n_derived = sum(1 for r in rows if r[1] == "DERIVED")
    n_mixed = sum(1 for r in rows if r[1] == "MIXED")
    n_config = sum(1 for r in rows if r[1].startswith("CONFIG"))
    n_pending = sum(1 for r in rows if r[1] == "NOT YET")

    # coverage stats
    def cov(df, col):
        if df is None or len(df) == 0 or col not in df.columns:
            return 0.0
        return float(df[col].notna().mean() * 100)

    lines = []
    lines.append("# Attribute Provenance Report — Project Dataset")
    lines.append("")
    lines.append("Generated by `scripts/build_project_dataset.py`.")
    lines.append("Schema target: PRD §10 (fields that real D3/TUM telemetry can support).")
    lines.append("")
    lines.append("## Counts by provenance tier")
    lines.append("")
    lines.append("| Tier | Count | Meaning |")
    lines.append("|---|---:|---|")
    lines.append(f"| **DATA** | {n_data} | Directly measured / recorded in source files |")
    lines.append(f"| **DERIVED** | {n_derived} | Computed from measured signals (unit convert, diffs, V×I) |")
    lines.append(f"| **MIXED** | {n_mixed} | DATA when present, else DERIVED fallback |")
    lines.append(f"| **CONFIG / ASSUMED** | {n_config} | Vehicle params — fill from EPA / `vehicles.yaml`, not telemetry |")
    lines.append(f"| **NOT YET** | {n_pending} | PRD labels needing physics engine + capacity (range, E_rem) |")
    lines.append("")
    total_ready = n_data + n_derived + n_mixed
    total = len(rows)
    lines.append(
        f"**Telemetry-ready now:** {total_ready}/{total} attributes "
        f"({100*total_ready/total:.0f}%). "
        f"**Still config/physics:** {n_config + n_pending}/{total} "
        f"({100*(n_config+n_pending)/total:.0f}%)."
    )
    lines.append("")
    lines.append("## Dataset build summary")
    lines.append("")
    lines.append("| Slice | Rows | Trips | Vehicles |")
    lines.append("|---|---:|---:|---|")
    lines.append(
        f"| Argonne D3 | {len(d3):,} | {d3['trip_id'].nunique() if len(d3) else 0} | "
        f"{', '.join(sorted(d3['vehicle_id'].unique())) if len(d3) else '-'} |"
    )
    lines.append(
        f"| TUM BMW i3 | {len(tum):,} | {tum['trip_id'].nunique() if len(tum) else 0} | "
        f"{', '.join(sorted(tum['vehicle_id'].unique())) if len(tum) else '-'} |"
    )
    lines.append(
        f"| Unified (1 Hz) | {len(unified):,} | {unified['trip_id'].nunique() if len(unified) else 0} | "
        f"{unified['vehicle_id'].nunique() if len(unified) else 0} vehicles |"
    )
    lines.append("")
    lines.append("### Notes / gaps")
    lines.append("")
    lines.append("- **2020 Tesla Model 3** zip is incomplete (`.part` only) — excluded.")
    lines.append("- **2023 Ioniq 5** Analysis.tdms has V/I (Hioki-like Udc1/Idc1) but **no SOC channel**.")
    lines.append("- **D3 grade is ~0%** (chassis dyno). Elevation term is dead for D3 — use TUM/MoST for grade.")
    lines.append("- **Range labels (range_a/b)** are not fabricated from dyno traces; they require the §6 physics + E14 pipeline.")
    lines.append("- Static vehicle parameters left NaN here on purpose — fill from EPA (`data/raw/epa/`) / `configs/vehicles.yaml`.")
    lines.append("")
    lines.append("## Attribute table")
    lines.append("")
    lines.append("| Attribute | Provenance | How obtained | Coverage D3 % | Coverage TUM % |")
    lines.append("|---|---|---|---:|---:|")
    for name, prov, how, _scope in rows:
        lines.append(
            f"| `{name}` | {prov} | {how} | {cov(d3, name):.1f} | {cov(tum, name):.1f} |"
        )
    lines.append("")
    lines.append("## Mapping to PRD §10")
    lines.append("")
    lines.append("| PRD field | Status in this dataset |")
    lines.append("|---|---|")
    lines.append("| Dynamic inputs (speed, accel, grade, elev, distance, stop, T_amb) | Present (grade/elev limited on D3) |")
    lines.append("| Static vehicle params | Columns reserved; values pending EPA fill |")
    lines.append("| p_batt, soc | Present where instrumented (derived p_batt from V×I) |")
    lines.append("| p_wheel | Present for D3 via F_trac×v |")
    lines.append("| energy_remaining, range_a, range_b | Deferred until physics engine + capacity |")
    lines.append("")
    lines.append("## Recommended next fills")
    lines.append("")
    lines.append("1. Join EPA-derived `mass_kg`, `C_rr`, `Cd·A` into static columns per `vehicle_id`.")
    lines.append("2. Set `battery_capacity_j` per vehicle; then `energy_remaining_j = soc * capacity`.")
    lines.append("3. Run `(E1)`–`(E14)` on MoST SUMO for the four training configs (Tier C).")
    lines.append("4. Use this D3+TUM set as **validation / sim-to-real**, not as the sole training set.")
    lines.append("")
    return "\n".join(lines)


def main():
    PROC.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    all_trips: list[pd.DataFrame] = []
    build_log = {"vehicles": {}}

    # Classic D3
    classic = [
        ("bmw_i3_2014", RAW_D3 / "2014 BMW i3"),
        ("kia_soul_ev_2015", RAW_D3 / "2015 Kia Soul EV"),
        ("ford_focus_ev_2013", RAW_D3 / "2013 Ford Focus Electric"),
        ("vw_egolf_2015", RAW_D3 / "2015 Volkswagen e-Golf"),
        ("mb_bclass_ev_2015", RAW_D3 / "2015 Mercedes Benz B-Class Electric"),
    ]
    for vid, folder in classic:
        if not folder.exists():
            build_log["vehicles"][vid] = {"status": "missing_folder"}
            continue
        trips = process_classic_vehicle(vid, folder)
        build_log["vehicles"][vid] = {"status": "ok", "trips": len(trips), "rows": int(sum(len(t) for t in trips))}
        all_trips.extend(trips)
        print(f"{vid}: {len(trips)} trips", flush=True)

    # Extended
    for vid, folder, kind in [
        ("nissan_leaf_eplus_2019", RAW_D3 / "2019 Nissan Leaf E Plus", "leaf"),
        ("chevy_bolt_2020", RAW_D3 / "2020 Chevrolet Bolt", "bolt"),
    ]:
        if not folder.exists():
            build_log["vehicles"][vid] = {"status": "missing_folder"}
            continue
        trips = process_extended_vehicle(vid, folder, kind)
        build_log["vehicles"][vid] = {"status": "ok", "trips": len(trips), "rows": int(sum(len(t) for t in trips))}
        all_trips.extend(trips)
        print(f"{vid}: {len(trips)} trips", flush=True)

    # Ioniq TDMS
    ioniq_dir = RAW_D3 / "2023 Hyundai Ioniq 5"
    if ioniq_dir.exists():
        trips = process_ioniq_tdms("hyundai_ioniq5_2023", ioniq_dir)
        build_log["vehicles"]["hyundai_ioniq5_2023"] = {
            "status": "ok_no_soc",
            "trips": len(trips),
            "rows": int(sum(len(t) for t in trips)),
        }
        all_trips.extend(trips)
        print(f"hyundai_ioniq5_2023: {len(trips)} trips", flush=True)

    build_log["vehicles"]["tesla_model3_2020"] = {
        "status": "skipped_incomplete_zip",
        "note": "2020 Tesla Model 3.hjTWwTNh.zip.part only",
    }

    d3 = pd.concat(all_trips, ignore_index=True) if all_trips else pd.DataFrame()
    print(f"D3 total rows (native Hz): {len(d3):,}", flush=True)

    tum_trips = process_tum()
    tum = pd.concat(tum_trips, ignore_index=True) if tum_trips else pd.DataFrame()
    build_log["vehicles"]["bmw_i3_tum"] = {
        "status": "ok",
        "trips": len(tum_trips),
        "rows": int(len(tum)),
    }
    print(f"TUM total rows: {len(tum):,}", flush=True)

    # 1 Hz versions for modelling
    d3_1hz = downsample_1hz(d3) if len(d3) else d3
    tum_1hz = downsample_1hz(tum) if len(tum) else tum
    unified = pd.concat([d3_1hz, tum_1hz], ignore_index=True) if len(d3_1hz) or len(tum_1hz) else pd.DataFrame()

    stats = {
        "d3_native": write_outputs(d3, "d3_native_hz"),
        "d3_1hz": write_outputs(d3_1hz, "d3_1hz"),
        "tum_native": write_outputs(tum, "tum_native_hz"),
        "tum_1hz": write_outputs(tum_1hz, "tum_1hz"),
        "project_unified_1hz": write_outputs(unified, "project_unified_1hz"),
        "build_log": build_log,
    }

    report = build_attribute_report(d3_1hz, tum_1hz, unified)
    (REP / "attribute_provenance.md").write_text(report)
    (REP / "build_summary.json").write_text(json.dumps(stats, indent=2, default=str))

    # short README in dataset/
    (OUT / "README.md").write_text(
        """# Project dataset (cleaned)

Built by `scripts/build_project_dataset.py`.

## Layout

- `processed/d3_1hz/` — Argonne D3, cleaned, ~1 Hz, per vehicle
- `processed/tum_1hz/` — TUM BMW i3 real trips, cleaned, ~1 Hz
- `processed/project_unified_1hz.csv.gz` — concatenated modelling table
- `reports/attribute_provenance.md` — **DATA vs DERIVED** attribute report
- `reports/build_summary.json` — row/trip counts

## Cleaning applied

- Dropped charge-only files
- Dropped t < 0 pre-buffers
- Dropped duplicate timestamps within a trip
- Dropped non-finite values; clipped accel and SOC
- SI units for speed (m/s), power (W), SOC (0–1)
- `p_batt_w` = V×I with discharge-positive convention

## Not included yet

- Tesla Model 3 (incomplete download)
- Static EPA vehicle params (join separately)
- Range labels (need physics engine)
"""
    )
    print("Wrote", REP / "attribute_provenance.md")
    print("Done.")


if __name__ == "__main__":
    main()
