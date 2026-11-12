"""Sliding-window sequence construction and training-only normalisation (PRD §14)."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DYNAMIC_FEATURES: tuple[str, ...] = (
    "speed_mps",
    "accel_mps2",
    "grade_rad",
    "elevation_m",
    "distance_m",
    "stop_flag",
    "ambient_temp_c",
)

STATIC_FEATURES: tuple[str, ...] = (
    "mass_kg",
    "drag_coefficient",
    "frontal_area_m2",
    "rolling_resistance",
    "battery_capacity_j",
    "drivetrain_efficiency",
    "regen_efficiency",
    "aux_power_base_w",
    "rotating_mass_factor",
)

TARGET_COLUMNS: tuple[str, ...] = (
    "soc",
    "p_batt_w",
    "range_b_m",
)


@dataclass
class ScalerBundle:
    """Separate StandardScalers fitted strictly on the training split (PRD §14.3, §22.7)."""

    dynamic_scaler: StandardScaler
    static_scaler: StandardScaler
    soc_scaler: StandardScaler
    power_scaler: StandardScaler
    range_scaler: StandardScaler
    n_train_rows: int
    w_soc: float
    w_pow: float
    w_rng: float

    def assert_fitted_on_train_only(self, expected_train_rows: int) -> None:
        """PRD §22.7: scaler fitted sample count must equal training split row count."""
        for name, sc in (
            ("dynamic", self.dynamic_scaler),
            ("static", self.static_scaler),
            ("soc", self.soc_scaler),
            ("power", self.power_scaler),
            ("range", self.range_scaler),
        ):
            seen = int(sc.n_samples_seen_)
            if seen != expected_train_rows:
                raise AssertionError(
                    f"{name} scaler n_samples_seen_={seen} != expected_train_rows={expected_train_rows}"
                )

    def target_stats(self) -> dict[str, tuple[float, float]]:
        """Return (mean, scale) for each target head in physical SI units."""
        return {
            "soc": (float(self.soc_scaler.mean_[0]), float(self.soc_scaler.scale_[0])),
            "p_batt_w": (float(self.power_scaler.mean_[0]), float(self.power_scaler.scale_[0])),
            "range_b_m": (float(self.range_scaler.mean_[0]), float(self.range_scaler.scale_[0])),
        }

    def head_weights_dict(self) -> dict[str, float]:
        """Return (L0) inverse-variance weights computed on the training split."""
        return {
            "w_soc": float(self.w_soc),
            "w_pow": float(self.w_pow),
            "w_rng": float(self.w_rng),
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh)
        return p

    @classmethod
    def load(cls, path: str | Path) -> ScalerBundle:
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected ScalerBundle, got {type(obj)}")
        return obj


def fit_scalers(train_df: pd.DataFrame, eps: float = 1e-12) -> ScalerBundle:
    """Fit separate StandardScalers on the training split rows only (PRD §12.2 L0, §14.3)."""
    if train_df.empty:
        raise ValueError("Cannot fit scalers on an empty training DataFrame.")

    n_rows = len(train_df)

    dyn_sc = StandardScaler().fit(train_df[list(DYNAMIC_FEATURES)].to_numpy(dtype=np.float64))
    stat_sc = StandardScaler().fit(train_df[list(STATIC_FEATURES)].to_numpy(dtype=np.float64))

    soc_vals = train_df[["soc"]].to_numpy(dtype=np.float64)
    pow_vals = train_df[["p_batt_w"]].to_numpy(dtype=np.float64)
    rng_mask = train_df["range_valid_mask"].to_numpy(dtype=bool)
    rng_all = train_df[["range_b_m"]].to_numpy(dtype=np.float64)

    soc_sc = StandardScaler().fit(soc_vals)
    pow_sc = StandardScaler().fit(pow_vals)
    rng_sc = StandardScaler().fit(rng_all)

    var_soc = float(np.var(soc_vals))
    var_pow = float(np.var(pow_vals))
    var_rng = float(np.var(rng_all[rng_mask])) if np.any(rng_mask) else float(np.var(rng_all))

    bundle = ScalerBundle(
        dynamic_scaler=dyn_sc,
        static_scaler=stat_sc,
        soc_scaler=soc_sc,
        power_scaler=pow_sc,
        range_scaler=rng_sc,
        n_train_rows=n_rows,
        w_soc=1.0 / max(var_soc, eps),
        w_pow=1.0 / max(var_pow, eps),
        w_rng=1.0 / max(var_rng, eps),
    )
    bundle.assert_fitted_on_train_only(n_rows)
    return bundle


@dataclass
class SequenceWindows:
    """Windowed tensor-ready arrays within trip boundaries (PRD §14.1)."""

    x_dyn: np.ndarray              # [N, T_seq, F_dyn] normalised
    x_stat: np.ndarray             # [N, F_stat] normalised
    soc: np.ndarray                # [N, T_seq] raw [0, 1]
    p_batt_w: np.ndarray           # [N, T_seq] raw Watts
    range_b_m: np.ndarray          # [N, T_seq] raw meters
    soc_norm: np.ndarray           # [N, T_seq] normalised
    p_batt_norm: np.ndarray        # [N, T_seq] normalised
    range_b_norm: np.ndarray       # [N, T_seq] normalised
    soc_mask: np.ndarray           # [N, T_seq] {0, 1}
    range_mask: np.ndarray         # [N, T_seq] {0, 1}
    e_usable_j: np.ndarray         # [N] raw Joules
    dt_s: np.ndarray               # [N] raw seconds
    speed_mps_raw: np.ndarray      # [N, T_seq] raw m/s
    accel_mps2_raw: np.ndarray     # [N, T_seq] raw m/s^2
    grade_rad_raw: np.ndarray      # [N, T_seq] raw rad
    ambient_temp_c_raw: np.ndarray # [N, T_seq] raw C
    trip_ids: list[str]
    vehicle_ids: list[str]
    start_idx: list[int] = field(default_factory=list)  # window start row within its trip

    def __len__(self) -> int:
        return int(self.x_dyn.shape[0])


def build_windows(
    df: pd.DataFrame,
    scalers: ScalerBundle,
    t_seq: int = 60,
    stride: int = 5,
) -> SequenceWindows:
    """Construct sliding windows `rows[i*S : i*S + T_seq]` strictly within each trip (PRD §14.1)."""
    if t_seq < 2:
        raise ValueError(f"t_seq must be >= 2 for L2 consistency residual, got {t_seq}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    x_dyn_list: list[np.ndarray] = []
    x_stat_list: list[np.ndarray] = []
    soc_list: list[np.ndarray] = []
    pow_list: list[np.ndarray] = []
    rng_list: list[np.ndarray] = []
    soc_n_list: list[np.ndarray] = []
    pow_n_list: list[np.ndarray] = []
    rng_n_list: list[np.ndarray] = []
    soc_mask_list: list[np.ndarray] = []
    rng_mask_list: list[np.ndarray] = []
    e_usable_list: list[float] = []
    dt_list: list[float] = []
    speed_raw_list: list[np.ndarray] = []
    accel_raw_list: list[np.ndarray] = []
    grade_raw_list: list[np.ndarray] = []
    temp_raw_list: list[np.ndarray] = []
    trip_ids: list[str] = []
    vehicle_ids: list[str] = []
    start_idx_list: list[int] = []

    for trip_id, group in df.groupby("trip_id", sort=False):
        trip = group.sort_values("timestamp", kind="mergesort")
        n = len(trip)
        if n < t_seq:
            continue

        dyn_raw = trip[list(DYNAMIC_FEATURES)].to_numpy(dtype=np.float64)
        stat_raw = trip[list(STATIC_FEATURES)].iloc[0:1].to_numpy(dtype=np.float64)

        dyn_norm = scalers.dynamic_scaler.transform(dyn_raw).astype(np.float32)
        stat_norm = scalers.static_scaler.transform(stat_raw)[0].astype(np.float32)

        soc_raw = trip["soc"].to_numpy(dtype=np.float32)
        pow_raw = trip["p_batt_w"].to_numpy(dtype=np.float32)
        rng_raw = trip["range_b_m"].to_numpy(dtype=np.float32)

        soc_norm = scalers.soc_scaler.transform(soc_raw.reshape(-1, 1)).ravel().astype(np.float32)
        pow_norm = scalers.power_scaler.transform(pow_raw.reshape(-1, 1)).ravel().astype(np.float32)
        rng_norm = scalers.range_scaler.transform(rng_raw.reshape(-1, 1)).ravel().astype(np.float32)

        s_mask = trip["soc_label_mask"].to_numpy(dtype=np.float32)
        r_mask = trip["range_valid_mask"].to_numpy(dtype=np.float32)

        ts = trip["timestamp"].to_numpy(dtype=np.float64)
        dt_val = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
        e_usable_val = float(trip["battery_capacity_j"].iloc[0])
        vid = str(trip["vehicle_id"].iloc[0])

        v_raw = trip["speed_mps"].to_numpy(dtype=np.float32)
        a_raw = trip["accel_mps2"].to_numpy(dtype=np.float32)
        g_raw = trip["grade_rad"].to_numpy(dtype=np.float32)
        t_raw = trip["ambient_temp_c"].to_numpy(dtype=np.float32)

        for start in range(0, n - t_seq + 1, stride):
            end = start + t_seq
            x_dyn_list.append(dyn_norm[start:end])
            x_stat_list.append(stat_norm)
            soc_list.append(soc_raw[start:end])
            pow_list.append(pow_raw[start:end])
            rng_list.append(rng_raw[start:end])
            soc_n_list.append(soc_norm[start:end])
            pow_n_list.append(pow_norm[start:end])
            rng_n_list.append(rng_norm[start:end])
            soc_mask_list.append(s_mask[start:end])
            rng_mask_list.append(r_mask[start:end])
            e_usable_list.append(e_usable_val)
            dt_list.append(dt_val)
            speed_raw_list.append(v_raw[start:end])
            accel_raw_list.append(a_raw[start:end])
            grade_raw_list.append(g_raw[start:end])
            temp_raw_list.append(t_raw[start:end])
            trip_ids.append(str(trip_id))
            vehicle_ids.append(vid)
            start_idx_list.append(int(start))

    if not x_dyn_list:
        raise ValueError(f"No windows of length t_seq={t_seq} could be built from DataFrame.")

    return SequenceWindows(
        x_dyn=np.stack(x_dyn_list, axis=0),
        x_stat=np.stack(x_stat_list, axis=0),
        soc=np.stack(soc_list, axis=0),
        p_batt_w=np.stack(pow_list, axis=0),
        range_b_m=np.stack(rng_list, axis=0),
        soc_norm=np.stack(soc_n_list, axis=0),
        p_batt_norm=np.stack(pow_n_list, axis=0),
        range_b_norm=np.stack(rng_n_list, axis=0),
        soc_mask=np.stack(soc_mask_list, axis=0),
        range_mask=np.stack(rng_mask_list, axis=0),
        e_usable_j=np.asarray(e_usable_list, dtype=np.float32),
        dt_s=np.asarray(dt_list, dtype=np.float32),
        speed_mps_raw=np.stack(speed_raw_list, axis=0),
        accel_mps2_raw=np.stack(accel_raw_list, axis=0),
        grade_rad_raw=np.stack(grade_raw_list, axis=0),
        ambient_temp_c_raw=np.stack(temp_raw_list, axis=0),
        trip_ids=trip_ids,
        vehicle_ids=vehicle_ids,
        start_idx=start_idx_list,
    )
