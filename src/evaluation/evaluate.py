"""Checkpoint + scaler loader and evaluation runner for baselines and neural models (PRD §15, §17)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.data.sequence_builder import ScalerBundle, SequenceWindows
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_soc_stratified_range_metrics,
)
from src.models.lstm import (
    AnalyticPhysicsBaseline,
    EVTrajectoryLSTM,
    RidgeWindowBaseline,
)
from src.models.pinn import build_model
from src.training.seed import resolve_device


def load_trained_checkpoint(
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    device: str = "cpu",
) -> tuple[EVTrajectoryLSTM, ScalerBundle, dict[str, Any]]:
    """Load a persisted model checkpoint and its paired `ScalerBundle` (PRD §14.3)."""
    dev = resolve_device(device)
    scalers = ScalerBundle.load(scaler_path)
    ckpt = torch.load(Path(checkpoint_path), map_location=dev, weights_only=False)

    cfg = ckpt.get("config", {})
    model_id = str(ckpt.get("model_id", cfg.get("model_id", "B")))
    hidden_dim = int(cfg.get("hidden_dim", 128))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.1))
    use_attention = bool(cfg.get("use_attention", model_id.upper() == "C"))

    model = build_model(
        model_id=model_id,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_attention_override=use_attention if model_id.upper() != "C" else True,
    ).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.set_target_scaler_stats(scalers.target_stats())
    model.eval()

    meta = {
        "run_id": ckpt.get("run_id"),
        "model_id": model_id,
        "best_epoch": ckpt.get("best_epoch"),
        "config": cfg,
    }
    return model, scalers, meta


@torch.no_grad()
def evaluate_neural_checkpoint(
    checkpoint_path: str | Path,
    scaler_path: str | Path,
    windows: SequenceWindows,
    device: str = "cpu",
) -> dict[str, Any]:
    """Evaluate a saved checkpoint on `windows`, returning overall and SOC-stratified metrics."""
    dev = resolve_device(device)
    model, scalers, meta = load_trained_checkpoint(checkpoint_path, scaler_path, device=device)

    x_dyn = torch.from_numpy(windows.x_dyn).to(dev)
    x_stat = torch.from_numpy(windows.x_stat).to(dev)
    preds = model(x_dyn, x_stat)

    soc_hat = preds["soc_hat"].cpu().numpy()
    p_hat_w = preds["p_hat_w"].cpu().numpy()
    range_hat_m = preds["range_hat_m"].cpu().numpy()

    metrics = compute_all_metrics(
        soc_hat=soc_hat,
        p_hat_w=p_hat_w,
        range_hat_m=range_hat_m,
        soc_true=windows.soc,
        p_true_w=windows.p_batt_w,
        range_true_m=windows.range_b_m,
        range_mask=windows.range_mask,
        dt_s=windows.dt_s,
        e_usable_j=windows.e_usable_j,
    )
    stratified = compute_soc_stratified_range_metrics(
        soc_true=windows.soc,
        range_hat_m=range_hat_m,
        range_true_m=windows.range_b_m,
        range_mask=windows.range_mask,
    )
    return {
        "meta": meta,
        "metrics": metrics,
        "soc_stratified": stratified,
        "predictions": {
            "soc_hat": soc_hat,
            "p_hat_w": p_hat_w,
            "range_hat_m": range_hat_m,
        },
    }


def evaluate_baselines(
    train_windows: SequenceWindows,
    test_windows: SequenceWindows,
    vehicles_yaml: str | Path,
    ridge_alpha: float = 1.0,
    test_trip_frames: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Evaluate Base-0 (Analytic Physics) and Base-1 (Ridge Window) on `test_windows` (PRD §15)."""
    base0 = AnalyticPhysicsBaseline(vehicles_yaml=vehicles_yaml)
    pred0 = base0.predict(test_windows, trip_frames=test_trip_frames)
    metrics0 = compute_all_metrics(
        soc_hat=pred0["soc_hat"],
        p_hat_w=pred0["p_hat_w"],
        range_hat_m=pred0["range_hat_m"],
        soc_true=test_windows.soc,
        p_true_w=test_windows.p_batt_w,
        range_true_m=test_windows.range_b_m,
        range_mask=test_windows.range_mask,
        dt_s=test_windows.dt_s,
        e_usable_j=test_windows.e_usable_j,
    )
    strat0 = compute_soc_stratified_range_metrics(
        soc_true=test_windows.soc,
        range_hat_m=pred0["range_hat_m"],
        range_true_m=test_windows.range_b_m,
        range_mask=test_windows.range_mask,
    )

    base1 = RidgeWindowBaseline(alpha=ridge_alpha).fit(train_windows)
    pred1 = base1.predict(test_windows)
    metrics1 = compute_all_metrics(
        soc_hat=pred1["soc_hat"],
        p_hat_w=pred1["p_hat_w"],
        range_hat_m=pred1["range_hat_m"],
        soc_true=test_windows.soc,
        p_true_w=test_windows.p_batt_w,
        range_true_m=test_windows.range_b_m,
        range_mask=test_windows.range_mask,
        dt_s=test_windows.dt_s,
        e_usable_j=test_windows.e_usable_j,
    )
    strat1 = compute_soc_stratified_range_metrics(
        soc_true=test_windows.soc,
        range_hat_m=pred1["range_hat_m"],
        range_true_m=test_windows.range_b_m,
        range_mask=test_windows.range_mask,
    )

    return {
        "Base-0": {
            "metrics": metrics0,
            "soc_stratified": strat0,
            "predictions": pred0,
        },
        "Base-1": {
            "metrics": metrics1,
            "soc_stratified": strat1,
            "predictions": pred1,
        },
    }
