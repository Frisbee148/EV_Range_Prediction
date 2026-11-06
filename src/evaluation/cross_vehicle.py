"""Leave-One-Vehicle-Out (LOVO / E5) cross-vehicle split generation and per-fold reporting (PRD §14.2, §16.5)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from src.data.splits import assert_no_leakage
from src.evaluation.metrics import aggregate_seed_metrics


def build_lovo_splits(
    df: pd.DataFrame,
    held_out_vehicle_id: str,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Construct a Leave-One-Vehicle-Out split where `held_out_vehicle_id` is strictly in `test` (PRD §14.2, §16.5).

    Within the remaining training vehicles, trips are stratified by `(vehicle_id, scenario)`
    into `train` and `val`.
    """
    meta = (
        df[["trip_id", "vehicle_id", "scenario"]]
        .drop_duplicates()
        .sort_values("trip_id")
    )
    all_vehicles = set(meta["vehicle_id"].astype(str).unique())
    if held_out_vehicle_id not in all_vehicles:
        raise ValueError(
            f"held_out_vehicle_id={held_out_vehicle_id!r} not in dataset vehicles {sorted(all_vehicles)}"
        )

    test_trips = sorted(
        meta[meta["vehicle_id"] == held_out_vehicle_id]["trip_id"].astype(str).unique()
    )
    dev_meta = meta[meta["vehicle_id"] != held_out_vehicle_id]

    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in dev_meta.itertuples(index=False):
        groups[(str(row.vehicle_id), str(row.scenario))].append(str(row.trip_id))

    train_trips: list[str] = []
    val_trips: list[str] = []
    for trips in groups.values():
        unique_trips = sorted(set(trips))
        rng.shuffle(unique_trips)
        n = len(unique_trips)
        n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
        val_trips.extend(unique_trips[:n_val])
        train_trips.extend(unique_trips[n_val:])

    split = {
        "train": sorted(set(train_trips)),
        "val": sorted(set(val_trips)),
        "test": test_trips,
    }
    assert_no_leakage(split)
    return split


def all_lovo_folds(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, list[str]]]:
    """Generate all four Leave-One-Vehicle-Out folds keyed by held-out `vehicle_id` (PRD §16.5)."""
    vehicles = sorted(df["vehicle_id"].astype(str).unique())
    return {
        vid: build_lovo_splits(df, held_out_vehicle_id=vid, val_frac=val_frac, seed=seed)
        for vid in vehicles
    }


def summarize_lovo_folds(
    fold_seed_metrics: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    """Aggregate LOVO results per held-out vehicle fold AND across folds (PRD §16.5, §26)."""
    if not fold_seed_metrics:
        raise ValueError("fold_seed_metrics cannot be empty.")

    per_fold: dict[str, dict[str, dict[str, Any]]] = {
        vid: aggregate_seed_metrics(seed_list)
        for vid, seed_list in fold_seed_metrics.items()
    }

    first_fold = next(iter(per_fold.values()))
    metric_keys = list(first_fold.keys())
    macro_avg: dict[str, dict[str, Any]] = {}
    for k in metric_keys:
        fold_means = np.asarray([per_fold[vid][k]["mean"] for vid in per_fold], dtype=np.float64)
        m_val = float(np.mean(fold_means))
        s_val = float(np.std(fold_means, ddof=0))
        prec = 6 if "residual" in k else 2
        macro_avg[k] = {
            "mean": m_val,
            "std_across_folds": s_val,
            "n_folds": len(fold_means),
            "formatted": f"{m_val:.{prec}f} ± {s_val:.{prec}f}",
        }

    return {
        "per_fold": per_fold,
        "macro_avg": macro_avg,
    }
