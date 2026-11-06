"""TDD test suite for Phase 5 Evaluation Framework (PRD §16.5, §16.10, §17, §22.8)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import load_trip_splits, prepare_datasets
from src.evaluation.cross_vehicle import (
    all_lovo_folds,
    build_lovo_splits,
    summarize_lovo_folds,
)
from src.evaluation.evaluate import (
    evaluate_baselines,
    evaluate_neural_checkpoint,
    load_trained_checkpoint,
)
from src.evaluation.metrics import (
    aggregate_seed_metrics,
    compute_all_metrics,
    compute_soc_stratified_range_metrics,
    format_results_table,
    paired_seed_comparison,
)
from src.evaluation.plots import (
    plot_data_ablation_curve,
    plot_lambda_sweep_curve,
    plot_loss_curves,
    plot_model_comparison_bars,
)
from src.training.trainer import Trainer, TrainerConfig
from src.vehicles import KWH_TO_J

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "sumo_most_1hz.csv.gz"
SPLITS_PATH = ROOT / "data" / "splits" / "trip_splits.json"
VEHICLES_PATH = ROOT / "configs" / "vehicles.yaml"


def test_metrics_and_soc_stratification():
    b, t = 4, 20
    soc_true = np.tile(np.linspace(0.95, 0.05, t, dtype=np.float32), (b, 1))
    soc_hat = np.clip(soc_true + 0.01, 0.0, 1.0)
    p_true = np.full((b, t), 15000.0, dtype=np.float32)
    p_hat = np.full((b, t), 15500.0, dtype=np.float32)
    r_true = np.tile(np.linspace(200e3, 10e3, t, dtype=np.float32), (b, 1))
    r_hat = r_true + 2000.0  # +2 km error everywhere
    r_mask = np.ones((b, t), dtype=np.float32)
    dt_s = np.ones(b, dtype=np.float32)
    e_usable = np.full(b, 50.0 * KWH_TO_J, dtype=np.float32)

    m = compute_all_metrics(
        soc_hat=soc_hat,
        p_hat_w=p_hat,
        range_hat_m=r_hat,
        soc_true=soc_true,
        p_true_w=p_true,
        range_true_m=r_true,
        range_mask=r_mask,
        dt_s=dt_s,
        e_usable_j=e_usable,
    )
    assert np.isclose(m["range_mae_km"], 2.0, atol=1e-4)
    assert np.isclose(m["soc_mae_pp"], 1.0, atol=1e-4)
    assert np.isclose(m["power_mae_kw"], 0.5, atol=1e-4)
    assert m["soc_bound_viol_pct"] == 0.0
    assert m["range_neg_viol_pct"] == 0.0

    strat = compute_soc_stratified_range_metrics(
        soc_true=soc_true,
        range_hat_m=r_hat,
        range_true_m=r_true,
        range_mask=r_mask,
    )
    assert set(strat.keys()) == {"100-75%", "75-50%", "50-25%", "25-0%"}
    for band_stats in strat.values():
        assert band_stats["count"] > 0
        assert np.isclose(band_stats["range_mae_km"], 2.0, atol=1e-4)


def test_seed_aggregation_and_paired_comparison():
    seeds_a = {
        s: {
            "range_mae_km": 12.0 + 0.2 * s,
            "range_rmse_km": 15.0 + 0.2 * s,
            "range_mape_pct": 8.0,
            "soc_mae_pp": 1.5,
            "power_mae_kw": 2.0,
            "residual_rms": 0.0020,
            "soc_bound_viol_pct": 0.5,
            "soc_mono_viol_pct": 1.2,
        }
        for s in range(5)
    }
    seeds_b = {
        s: {
            "range_mae_km": 10.5 + 0.2 * s,
            "range_rmse_km": 13.5 + 0.2 * s,
            "range_mape_pct": 7.0,
            "soc_mae_pp": 1.1,
            "power_mae_kw": 1.9,
            "residual_rms": 0.0001,
            "soc_bound_viol_pct": 0.0,
            "soc_mono_viol_pct": 0.1,
        }
        for s in range(5)
    }

    agg_a = aggregate_seed_metrics(list(seeds_a.values()))
    agg_b = aggregate_seed_metrics(list(seeds_b.values()))
    assert np.isclose(agg_a["range_mae_km"]["mean"], 12.4)
    assert np.isclose(agg_b["range_mae_km"]["mean"], 10.9)

    paired = paired_seed_comparison(seeds_a, seeds_b, metric_key="range_mae_km")
    assert paired["n_seeds"] == 5
    assert np.isclose(paired["mean_diff"], 1.5, atol=1e-6)
    assert np.isclose(paired["std_diff"], 0.0, atol=1e-6)

    table_md = format_results_table({"A": agg_a, "B": agg_b})
    assert "| Model |" in table_md
    assert "| A |" in table_md
    assert "| B |" in table_md


def test_lovo_cross_vehicle_splits_have_zero_vehicle_leakage():
    df = pd.read_csv(DATASET_PATH)
    folds = all_lovo_folds(df, val_frac=0.15, seed=42)
    assert set(folds.keys()) == {
        "bmw_i3_60ah",
        "ford_focus_ev_2016",
        "tesla_model_x_75d_2016",
        "chevy_bolt_ev_2023",
    }
    for held_out_vid, split in folds.items():
        train_vids = set(df[df["trip_id"].isin(set(split["train"]))]["vehicle_id"].unique())
        val_vids = set(df[df["trip_id"].isin(set(split["val"]))]["vehicle_id"].unique())
        test_vids = set(df[df["trip_id"].isin(set(split["test"]))]["vehicle_id"].unique())
        assert held_out_vid not in train_vids
        assert held_out_vid not in val_vids
        assert test_vids == {held_out_vid}

    summary = summarize_lovo_folds(
        {
            vid: [{"range_mae_km": 10.0 + i, "residual_rms": 0.001} for i in range(5)]
            for vid in folds
        }
    )
    assert set(summary["per_fold"].keys()) == set(folds.keys())
    assert "macro_avg" in summary


def test_checkpoint_loading_and_baselines_evaluation(tmp_path: Path):
    df = pd.read_csv(DATASET_PATH)
    full_splits = load_trip_splits(SPLITS_PATH)
    small_splits = {
        "train": full_splits["train"][:6],
        "val": full_splits["val"][:3],
        "test": full_splits["test"][:3],
    }
    sub_df = df[
        df["trip_id"].isin(
            set(small_splits["train"]) | set(small_splits["val"]) | set(small_splits["test"])
        )
    ].copy()

    prepared = prepare_datasets(sub_df, small_splits, t_seq=60, stride=15)
    cfg = TrainerConfig(
        model_id="B",
        T_seq=60,
        stride=15,
        batch_size=32,
        hidden_dim=32,
        num_layers=2,
        epochs=2,
        warmup_epochs=1,
        seed=0,
        device="cpu",
    )
    trainer = Trainer(cfg, prepared, root_dir=tmp_path)
    res = trainer.fit(run_id="eval_test_B", save_artifacts=True)

    loaded_model, loaded_scalers, meta = load_trained_checkpoint(
        res.checkpoint_path, res.scaler_path, device="cpu"
    )
    assert meta["model_id"] == "B"
    assert loaded_scalers.n_train_rows == prepared.scalers.n_train_rows

    eval_out = evaluate_neural_checkpoint(
        res.checkpoint_path,
        res.scaler_path,
        prepared.test_dataset.windows,
        device="cpu",
    )
    assert np.isclose(
        eval_out["metrics"]["range_mae_km"], res.test_metrics["range_mae_km"], atol=1e-4
    )
    assert "soc_stratified" in eval_out

    base_out = evaluate_baselines(
        train_windows=prepared.train_dataset.windows,
        test_windows=prepared.test_dataset.windows,
        vehicles_yaml=VEHICLES_PATH,
    )
    assert "Base-0" in base_out and "Base-1" in base_out
    assert base_out["Base-0"]["metrics"]["soc_mae_pp"] < 1e-3
    assert np.isfinite(base_out["Base-1"]["metrics"]["range_mae_km"])


def test_plotting_refuses_fewer_than_three_seeds_and_renders_valid_runs(tmp_path: Path):
    """PRD §22.8: plotting code must refuse comparison charts when fewer than 3 seeds are present."""
    too_few = {
        "A": [{"range_mae_km": 15.0}, {"range_mae_km": 16.0}],
        "B": [{"range_mae_km": 12.0}, {"range_mae_km": 13.0}],
    }
    with pytest.raises(ValueError, match="at least 3 seeds"):
        plot_model_comparison_bars(too_few, tmp_path / "bad_bars.png")

    valid_bars = {
        "A": [{"range_mae_km": 15.0 + 0.1 * s} for s in range(5)],
        "B": [{"range_mae_km": 12.0 + 0.1 * s} for s in range(5)],
    }
    bar_path = plot_model_comparison_bars(valid_bars, tmp_path / "bars.png")
    assert bar_path.exists()

    ablation_data = {
        0.10: {
            "A": [{"range_mae_km": 25.0 + s} for s in range(5)],
            "B": [{"range_mae_km": 18.0 + s} for s in range(5)],
        },
        0.50: {
            "A": [{"range_mae_km": 14.0 + 0.5 * s} for s in range(5)],
            "B": [{"range_mae_km": 12.5 + 0.5 * s} for s in range(5)],
        },
        1.00: {
            "A": [{"range_mae_km": 10.2 + 0.2 * s} for s in range(5)],
            "B": [{"range_mae_km": 10.0 + 0.2 * s} for s in range(5)],
        },
    }
    ablation_path = plot_data_ablation_curve(ablation_data, tmp_path / "ablation.png")
    assert ablation_path.exists()

    sweep_data = {
        1e-3: [{"range_mae_km": 14.0 + 0.1 * s, "residual_rms": 0.002} for s in range(3)],
        1e-1: [{"range_mae_km": 11.5 + 0.1 * s, "residual_rms": 0.0004} for s in range(3)],
        1e0: [{"range_mae_km": 11.0 + 0.1 * s, "residual_rms": 0.0001} for s in range(3)],
    }
    sweep_path = plot_lambda_sweep_curve(sweep_data, tmp_path / "lambda_sweep.png")
    assert sweep_path.exists()

    loss_path = plot_loss_curves(
        [
            {"epoch": 0, "train_loss_total": 3.0, "val_loss_total": 2.8, "train_loss_data": 3.0, "val_loss_data": 2.8},
            {"epoch": 1, "train_loss_total": 2.1, "val_loss_total": 2.0, "train_loss_data": 1.9, "val_loss_data": 1.9},
        ],
        tmp_path / "loss.png",
    )
    assert loss_path.exists()
