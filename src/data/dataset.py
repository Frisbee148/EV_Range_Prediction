"""PyTorch Dataset and DataLoader factory for multi-vehicle EV trajectories (PRD §14)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.degradation import (
    apply_sensor_noise,
    apply_sparse_soc_mask,
    subsample_training_trips,
)
from src.data.sequence_builder import (
    ScalerBundle,
    SequenceWindows,
    build_windows,
    fit_scalers,
)
from src.data.splits import assert_no_leakage


class EVSequenceDataset(Dataset):
    """PyTorch Dataset wrapping a `SequenceWindows` container."""

    def __init__(self, windows: SequenceWindows) -> None:
        self.windows = windows
        self.x_dyn = torch.from_numpy(windows.x_dyn)
        self.x_stat = torch.from_numpy(windows.x_stat)
        self.soc = torch.from_numpy(windows.soc)
        self.p_batt_w = torch.from_numpy(windows.p_batt_w)
        self.range_b_m = torch.from_numpy(windows.range_b_m)
        self.soc_norm = torch.from_numpy(windows.soc_norm)
        self.p_batt_norm = torch.from_numpy(windows.p_batt_norm)
        self.range_b_norm = torch.from_numpy(windows.range_b_norm)
        self.soc_mask = torch.from_numpy(windows.soc_mask)
        self.range_mask = torch.from_numpy(windows.range_mask)
        self.e_usable_j = torch.from_numpy(windows.e_usable_j)
        self.dt_s = torch.from_numpy(windows.dt_s)
        self.speed_mps_raw = torch.from_numpy(windows.speed_mps_raw)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_dyn": self.x_dyn[idx],
            "x_stat": self.x_stat[idx],
            "soc": self.soc[idx],
            "p_batt_w": self.p_batt_w[idx],
            "range_b_m": self.range_b_m[idx],
            "soc_norm": self.soc_norm[idx],
            "p_batt_norm": self.p_batt_norm[idx],
            "range_b_norm": self.range_b_norm[idx],
            "soc_mask": self.soc_mask[idx],
            "range_mask": self.range_mask[idx],
            "e_usable_j": self.e_usable_j[idx],
            "dt_s": self.dt_s[idx],
            "speed_mps_raw": self.speed_mps_raw[idx],
        }


@dataclass
class PreparedSplits:
    """Bundled datasets, windows, and scalers for a training/evaluation run."""

    train_dataset: EVSequenceDataset
    val_dataset: EVSequenceDataset
    test_dataset: EVSequenceDataset
    scalers: ScalerBundle
    split_ids: dict[str, list[str]]


def load_trip_splits(path: str | Path) -> dict[str, list[str]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        split = json.load(fh)
    assert_no_leakage(split)
    return {k: list(v) for k, v in split.items()}


def prepare_datasets(
    df: pd.DataFrame,
    split_ids: dict[str, list[str]],
    *,
    t_seq: int = 60,
    stride: int = 5,
    ablation_fraction: float = 1.0,
    soc_k: int = 1,
    noise_enabled: bool = False,
    noise_seed: int = 42,
    ablation_seed: int = 0,
) -> PreparedSplits:
    """Split by trip, apply training degradations, fit scalers on train only, and window."""
    assert_no_leakage(split_ids)

    train_trips = subsample_training_trips(
        df, split_ids["train"], fraction=ablation_fraction, seed=ablation_seed
    )
    active_splits = {
        "train": train_trips,
        "val": list(split_ids["val"]),
        "test": list(split_ids["test"]),
    }
    assert_no_leakage(active_splits)

    train_df = df[df["trip_id"].isin(set(active_splits["train"]))].copy()
    val_df = df[df["trip_id"].isin(set(active_splits["val"]))].copy()
    test_df = df[df["trip_id"].isin(set(active_splits["test"]))].copy()

    # Apply sparse SOC mask and optional sensor noise
    train_df = apply_sparse_soc_mask(train_df, k=soc_k)
    if noise_enabled:
        train_df = apply_sensor_noise(train_df, enabled=True, noise_seed=noise_seed)
        val_df = apply_sensor_noise(val_df, enabled=True, noise_seed=noise_seed + 1)

    # Fit scalers strictly on training rows (PRD §14.3, §22.7)
    scalers = fit_scalers(train_df)
    scalers.assert_fitted_on_train_only(len(train_df))

    train_win = build_windows(train_df, scalers, t_seq=t_seq, stride=stride)
    val_win = build_windows(val_df, scalers, t_seq=t_seq, stride=stride)
    test_win = build_windows(test_df, scalers, t_seq=t_seq, stride=stride)

    # Verify window-level trip sets are disjoint
    assert_no_leakage(
        {
            "train": train_win.trip_ids,
            "val": val_win.trip_ids,
            "test": test_win.trip_ids,
        }
    )

    return PreparedSplits(
        train_dataset=EVSequenceDataset(train_win),
        val_dataset=EVSequenceDataset(val_win),
        test_dataset=EVSequenceDataset(test_win),
        scalers=scalers,
        split_ids=active_splits,
    )


def build_dataloaders(
    prepared: PreparedSplits,
    batch_size: int = 256,
    seed: int = 0,
) -> dict[str, DataLoader]:
    """Create deterministic train, val, and test DataLoaders."""
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return {
        "train": DataLoader(
            prepared.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=gen,
            drop_last=False,
        ),
        "val": DataLoader(
            prepared.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        ),
        "test": DataLoader(
            prepared.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        ),
    }
