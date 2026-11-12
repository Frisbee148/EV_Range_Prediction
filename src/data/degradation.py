"""Information-gap degradation mechanisms (PRD §11).

Implements:
- §11.1 Data ablation (trip-level subsampling of training trips)
- §11.2 Sparse SOC supervision (supervise SOC only every k timesteps)
- §11.3 Sensor noise (additive Gaussian noise on inputs + quantised/noisy SOC label)
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

SPEED_NOISE_STD_MPS = 0.3
ACCEL_NOISE_STD_MPS2 = 0.15
GRADE_NOISE_STD_RAD = 0.005
SOC_QUANT_STEP = 0.01
SOC_NOISE_STD = 0.005


def subsample_training_trips(
    df: pd.DataFrame,
    train_trip_ids: list[str],
    fraction: float,
    seed: int = 0,
) -> list[str]:
    """Sample `fraction` of `train_trip_ids` at trip granularity (PRD §11.1).

    The total number of trips is `round(fraction * N)` (at least 1). It is split
    across `(vehicle_id, scenario)` groups in proportion to group size using the
    largest-remainder method, so the requested fractions give distinct, honest
    subset sizes. A forced "at least one trip per group" floor is NOT applied:
    it made the 5% and 10% settings select the same trips. At very small
    fractions some groups can therefore have no training trips.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"ablation fraction must be in (0, 1], got {fraction}")
    if fraction >= 1.0:
        return sorted(set(train_trip_ids))

    train_set = set(train_trip_ids)
    meta = (
        df[df["trip_id"].isin(train_set)][["trip_id", "vehicle_id", "scenario"]]
        .drop_duplicates()
        .sort_values("trip_id")
    )
    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in meta.itertuples(index=False):
        groups[(str(row.vehicle_id), str(row.scenario))].append(str(row.trip_id))

    keys = sorted(groups)
    sizes = {key: len(set(groups[key])) for key in keys}
    n_all = sum(sizes.values())
    n_total = min(n_all, max(1, int(round(n_all * fraction))))

    quotas = {key: sizes[key] * fraction for key in keys}
    alloc = {key: min(sizes[key], int(np.floor(quotas[key]))) for key in keys}
    tie_break = {key: float(rng.random()) for key in keys}
    by_remainder = sorted(
        keys, key=lambda key: (quotas[key] - np.floor(quotas[key]), tie_break[key]), reverse=True
    )
    remaining = n_total - sum(alloc.values())
    while remaining > 0:
        progressed = False
        for key in by_remainder:
            if remaining == 0:
                break
            if alloc[key] < sizes[key]:
                alloc[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break

    selected: list[str] = []
    for key in keys:
        trips_sorted = sorted(set(groups[key]))
        rng.shuffle(trips_sorted)
        selected.extend(trips_sorted[: alloc[key]])

    return sorted(set(selected))


def apply_sparse_soc_mask(df: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """Supervise SOC only every `k` timesteps within each trip (PRD §11.2)."""
    if k < 1:
        raise ValueError(f"SOC supervision interval k must be >= 1, got {k}")
    out = df.copy()
    if k == 1:
        out["soc_label_mask"] = 1
        return out

    masks = np.zeros(len(out), dtype=np.int64)
    for _, group in out.groupby("trip_id", sort=False):
        idx = group.index.to_numpy()
        step_idx = np.arange(len(idx))
        masks[out.index.get_indexer(idx)] = (step_idx % k == 0).astype(np.int64)
    out["soc_label_mask"] = masks
    return out


def apply_sensor_noise(
    df: pd.DataFrame,
    enabled: bool = False,
    noise_seed: int = 42,
) -> pd.DataFrame:
    """Apply sensor noise to features and SOC labels only (PRD §11.3)."""
    if not enabled:
        return df
    out = df.copy()
    rng = np.random.default_rng(noise_seed)
    n = len(out)

    out["speed_mps"] = np.maximum(
        0.0,
        out["speed_mps"].to_numpy(dtype=float) + rng.normal(0.0, SPEED_NOISE_STD_MPS, size=n),
    )
    out["accel_mps2"] = out["accel_mps2"].to_numpy(dtype=float) + rng.normal(
        0.0, ACCEL_NOISE_STD_MPS2, size=n
    )
    out["grade_rad"] = out["grade_rad"].to_numpy(dtype=float) + rng.normal(
        0.0, GRADE_NOISE_STD_RAD, size=n
    )
    soc_raw = out["soc"].to_numpy(dtype=float)
    soc_quant = np.round(soc_raw / SOC_QUANT_STEP) * SOC_QUANT_STEP
    soc_noisy = np.clip(soc_quant + rng.normal(0.0, SOC_NOISE_STD, size=n), 0.0, 1.0)
    out["soc"] = soc_noisy
    out["noise_seed"] = int(noise_seed)
    return out
