"""Trip-level splits with leakage checks (PRD §14.2 / §22.5)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


def trip_level_split(
    trip_keys: pd.DataFrame,
    rng: np.random.Generator,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, list[str]]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in trip_keys.itertuples(index=False):
        groups[(str(row.vehicle_id), str(row.scenario))].append(str(row.trip_id))
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for trips in groups.values():
        trips = list(dict.fromkeys(trips))
        rng.shuffle(trips)
        n = len(trips)
        if n == 1:
            split["train"].append(trips[0])
            continue
        if n == 2:
            split["train"].append(trips[0])
            split["test"].append(trips[1])
            continue
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac))
        while n_train + n_val >= n:
            if n_val > 1:
                n_val -= 1
            elif n_train > 1:
                n_train -= 1
            else:
                break
        split["train"].extend(trips[:n_train])
        split["val"].extend(trips[n_train : n_train + n_val])
        split["test"].extend(trips[n_train + n_val :])
    return {k: sorted(set(v)) for k, v in split.items()}


def assert_no_leakage(split: dict[str, list[str]]) -> None:
    sets = {k: set(v) for k, v in split.items()}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
