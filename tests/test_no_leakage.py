"""Trip-level split leakage (PRD §22.5)."""

import numpy as np
import pandas as pd

from src.data.splits import assert_no_leakage, trip_level_split


def test_splits_are_disjoint():
    keys = []
    for v in ("a", "b"):
        for s in ("urban", "highway"):
            for i in range(10):
                keys.append({"trip_id": f"{v}_{s}_{i}", "vehicle_id": v, "scenario": s})
    split = trip_level_split(pd.DataFrame(keys), np.random.default_rng(0))
    assert_no_leakage(split)
    all_ids = set(split["train"]) | set(split["val"]) | set(split["test"])
    assert all_ids == {r["trip_id"] for r in keys}
    assert split["train"] and split["val"] and split["test"]
