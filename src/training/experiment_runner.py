"""Orchestrator for PRD §16 Experiments E1–E9 (PRD §16, §17, §20)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.dataset import load_trip_splits, prepare_datasets
from src.data.splits import assert_no_leakage
from src.evaluation.cross_vehicle import all_lovo_folds, summarize_lovo_folds
from src.evaluation.evaluate import evaluate_baselines
from src.evaluation.metrics import (
    aggregate_seed_metrics,
    format_results_table,
    paired_seed_comparison,
)
from src.evaluation.plots import (
    plot_data_ablation_curve,
    plot_lambda_sweep_curve,
    plot_model_comparison_bars,
)
from src.physics.trip import simulate_trip
from src.training.trainer import Trainer, TrainerConfig
from src.vehicles import load_all_vehicles, load_physics_constants


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "experiment_id" not in cfg:
        raise ValueError(f"Experiment YAML {path} missing 'experiment_id'")
    return cfg


def build_cross_regime_splits(
    df: pd.DataFrame,
    train_scenarios: tuple[str, ...] = ("urban", "arterial"),
    test_scenarios: tuple[str, ...] = ("highway",),
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Construct E6 Cross-Regime splits: train/val on `train_scenarios`, test on `test_scenarios` (PRD §16.6)."""
    meta = (
        df[["trip_id", "vehicle_id", "scenario"]]
        .drop_duplicates()
        .sort_values("trip_id")
    )
    test_trips = sorted(
        meta[meta["scenario"].isin(set(test_scenarios))]["trip_id"].astype(str).unique()
    )
    dev_meta = meta[meta["scenario"].isin(set(train_scenarios))]

    rng = np.random.default_rng(seed)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in dev_meta.itertuples(index=False):
        groups[(str(row.vehicle_id), str(row.scenario))].append(str(row.trip_id))

    train_trips: list[str] = []
    val_trips: list[str] = []
    for trips in groups.values():
        u_trips = sorted(set(trips))
        rng.shuffle(u_trips)
        n_val = max(1, int(round(len(u_trips) * val_frac))) if len(u_trips) > 1 else 0
        val_trips.extend(u_trips[:n_val])
        train_trips.extend(u_trips[n_val:])

    split = {
        "train": sorted(set(train_trips)),
        "val": sorted(set(val_trips)),
        "test": test_trips,
    }
    assert_no_leakage(split)
    return split


def run_parameter_sensitivity_analysis(
    df: pd.DataFrame,
    vehicles_yaml: str | Path,
) -> dict[str, dict[str, float]]:
    """E9 Parameter Sensitivity: regenerate physics labels at low/mid/high [ASSUMED] bounds (PRD §16.9).

    Tiers (from PRD §8.2):
    - `low` drag/losses (`C_rr = 0.008`, `eta_dt = 0.93`) -> upper range bound
    - `mid` (`C_rr = 0.010`, `eta_dt = 0.90`) -> nominal midpoint
    - `high` drag/losses (`C_rr = 0.012`, `eta_dt = 0.85`) -> lower range bound
    """
    vehicles = load_all_vehicles(vehicles_yaml)
    constants = load_physics_constants(vehicles_yaml)
    tiers = {
        "low": {"rolling_resistance": 0.008, "eta_dt": 0.93},
        "mid": {"rolling_resistance": 0.010, "eta_dt": 0.90},
        "high": {"rolling_resistance": 0.012, "eta_dt": 0.85},
    }

    results: dict[str, dict[str, float]] = {}
    for tier_name, params in tiers.items():
        range_b_vals: list[np.ndarray] = []
        p_batt_vals: list[np.ndarray] = []
        soc_end_vals: list[float] = []

        for _, group in df.groupby("trip_id", sort=False):
            trip = group.sort_values("timestamp")
            vid = str(trip["vehicle_id"].iloc[0])
            v_mod = replace(
                vehicles[vid],
                rolling_resistance=params["rolling_resistance"],
                eta_dt=params["eta_dt"],
            )
            ts = trip["timestamp"].to_numpy(dtype=float)
            dt_s = float(np.median(np.diff(ts))) if len(ts) > 1 else 1.0
            sim = simulate_trip(
                v_mps=trip["speed_mps"].to_numpy(dtype=float),
                a_mps2=trip["accel_mps2"].to_numpy(dtype=float),
                theta_rad=trip["grade_rad"].to_numpy(dtype=float),
                t_amb_c=trip["ambient_temp_c"].to_numpy(dtype=float),
                soc_0=float(trip["soc"].iloc[0]),
                dt_s=dt_s,
                vehicle=v_mod,
                constants=constants,
            )
            range_b_vals.append(sim["range_b_m"] / 1000.0)
            p_batt_vals.append(sim["p_batt_w"] / 1000.0)
            soc_end_vals.append(float(sim["soc"][-1]))

        all_rng_km = np.concatenate(range_b_vals)
        all_pow_kw = np.concatenate(p_batt_vals)
        results[tier_name] = {
            "c_rr": float(params["rolling_resistance"]),
            "eta_dt": float(params["eta_dt"]),
            "mean_range_b_km": float(np.mean(all_rng_km)),
            "std_range_b_km": float(np.std(all_rng_km)),
            "mean_p_batt_kw": float(np.mean(all_pow_kw)),
            "mean_end_soc": float(np.mean(soc_end_vals)),
        }

    return results


def _write_outputs(
    root_dir: Path,
    experiment_id: str,
    payload: dict[str, Any],
    markdown_text: str,
) -> tuple[Path, Path]:
    metrics_dir = root_dir / "results" / "metrics"
    tables_dir = root_dir / "results" / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    json_path = metrics_dir / f"{experiment_id}.json"
    md_path = tables_dir / f"{experiment_id}.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(markdown_text, encoding="utf-8")
    return json_path, md_path


def run_experiment_suite(
    experiment_id: str,
    df: pd.DataFrame,
    split_ids: dict[str, list[str]],
    *,
    root_dir: str | Path,
    vehicles_yaml: str | Path,
    seeds: list[int] | None = None,
    epochs: int = 100,
    warmup_epochs: int = 10,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    t_seq: int = 60,
    stride: int = 5,
    lambda_max: float = 1.0,
    ablation_fractions: list[float] | None = None,
    soc_k_intervals: list[int] | None = None,
    lambda_grid: list[float] | None = None,
    t_seq_grid: list[int] | None = None,
    vehicles_subset: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a PRD §16 experiment (`E1`–`E9`) and persist metrics, tables, and plots."""
    exp_key = experiment_id.upper()
    root = Path(root_dir)
    plots_dir = root / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if exp_key == "E1":
        run_seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]
        prepared = prepare_datasets(df, split_ids, t_seq=t_seq, stride=stride)
        base_eval = evaluate_baselines(
            prepared.train_dataset.windows,
            prepared.test_dataset.windows,
            vehicles_yaml=vehicles_yaml,
        )

        seeds_by_model: dict[str, dict[int, dict[str, float]]] = {"A": {}, "B": {}}
        for m_id in ("A", "B"):
            for s in run_seeds:
                cfg = TrainerConfig(
                    model_id=m_id,
                    T_seq=t_seq,
                    stride=stride,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    warmup_epochs=warmup_epochs,
                    lambda_max=lambda_max,
                    seed=s,
                )
                res = Trainer(cfg, prepared, root_dir=root).fit(
                    run_id=f"E1_model_{m_id}_seed{s}", save_artifacts=True
                )
                seeds_by_model[m_id][s] = res.test_metrics

        agg_map = {
            "Base-0": aggregate_seed_metrics([base_eval["Base-0"]["metrics"]]),
            "Base-1": aggregate_seed_metrics([base_eval["Base-1"]["metrics"]]),
            "A": aggregate_seed_metrics(list(seeds_by_model["A"].values())),
            "B": aggregate_seed_metrics(list(seeds_by_model["B"].values())),
        }
        paired = paired_seed_comparison(seeds_by_model["A"], seeds_by_model["B"], "range_mae_km")
        table_md = f"# E1 In-Distribution Baseline\n\n{format_results_table(agg_map)}\n\n"
        table_md += f"**Paired Difference (MAE_A - MAE_B):** `{paired['formatted']}` km (n={paired['n_seeds']} seeds)\n"

        if len(run_seeds) >= 3:
            plot_model_comparison_bars(
                {
                    "A": list(seeds_by_model["A"].values()),
                    "B": list(seeds_by_model["B"].values()),
                },
                plots_dir / "E1_comparison_bars.png",
            )

        payload = {
            "experiment_id": "E1",
            "aggregated": agg_map,
            "paired_A_minus_B": paired,
            "table_markdown": table_md,
        }
        _write_outputs(root, "E1", payload, table_md)
        return payload

    if exp_key == "E2":
        run_seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]
        fracs = ablation_fractions if ablation_fractions is not None else [0.05, 0.10, 0.25, 0.50, 1.00]
        ablation_runs: dict[float, dict[str, list[dict[str, float]]]] = {}
        summary_rows: dict[str, dict[str, Any]] = {}

        for frac in fracs:
            ablation_runs[frac] = {"A": [], "B": []}
            for s in run_seeds:
                prepared = prepare_datasets(
                    df,
                    split_ids,
                    t_seq=t_seq,
                    stride=stride,
                    ablation_fraction=frac,
                    ablation_seed=s,
                )
                for m_id in ("A", "B"):
                    cfg = TrainerConfig(
                        model_id=m_id,
                        T_seq=t_seq,
                        stride=stride,
                        batch_size=batch_size,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        warmup_epochs=warmup_epochs,
                        lambda_max=lambda_max,
                        ablation_fraction=frac,
                        seed=s,
                    )
                    res = Trainer(cfg, prepared, root_dir=root).fit(
                        run_id=f"E2_frac{int(frac*100)}_model_{m_id}_seed{s}",
                        save_artifacts=True,
                    )
                    ablation_runs[frac][m_id].append(res.test_metrics)

            summary_rows[f"A ({int(frac*100)}%)"] = aggregate_seed_metrics(ablation_runs[frac]["A"])
            summary_rows[f"B ({int(frac*100)}%)"] = aggregate_seed_metrics(ablation_runs[frac]["B"])

        if len(run_seeds) >= 3:
            plot_data_ablation_curve(ablation_runs, plots_dir / "E2_data_ablation.png")

        table_md = f"# E2 Data Ablation\n\n{format_results_table(summary_rows)}\n"
        payload = {
            "experiment_id": "E2",
            "fractions": [float(f) for f in fracs],
            "aggregated": summary_rows,
            "table_markdown": table_md,
        }
        _write_outputs(root, "E2", payload, table_md)
        return payload

    if exp_key == "E3":
        run_seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]
        k_list = soc_k_intervals if soc_k_intervals is not None else [1, 10, 60, 300]
        summary_rows = {}
        for k_val in k_list:
            prepared = prepare_datasets(df, split_ids, t_seq=t_seq, stride=stride, soc_k=k_val)
            for m_id in ("A", "B"):
                runs = []
                for s in run_seeds:
                    cfg = TrainerConfig(
                        model_id=m_id,
                        T_seq=t_seq,
                        stride=stride,
                        batch_size=batch_size,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        warmup_epochs=warmup_epochs,
                        lambda_max=lambda_max,
                        soc_k=k_val,
                        seed=s,
                    )
                    res = Trainer(cfg, prepared, root_dir=root).fit(
                        run_id=f"E3_k{k_val}_model_{m_id}_seed{s}", save_artifacts=True
                    )
                    runs.append(res.test_metrics)
                summary_rows[f"{m_id} (k={k_val})"] = aggregate_seed_metrics(runs)

        table_md = f"# E3 Sparse SOC Supervision\n\n{format_results_table(summary_rows)}\n"
        payload = {"experiment_id": "E3", "aggregated": summary_rows, "table_markdown": table_md}
        _write_outputs(root, "E3", payload, table_md)
        return payload

    if exp_key == "E4":
        run_seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]
        summary_rows = {}
        for noisy in (False, True):
            prepared = prepare_datasets(
                df, split_ids, t_seq=t_seq, stride=stride, noise_enabled=noisy
            )
            label = "noisy" if noisy else "clean"
            for m_id in ("A", "B"):
                runs = []
                for s in run_seeds:
                    cfg = TrainerConfig(
                        model_id=m_id,
                        T_seq=t_seq,
                        stride=stride,
                        batch_size=batch_size,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        warmup_epochs=warmup_epochs,
                        lambda_max=lambda_max,
                        noise_enabled=noisy,
                        seed=s,
                    )
                    res = Trainer(cfg, prepared, root_dir=root).fit(
                        run_id=f"E4_{label}_model_{m_id}_seed{s}", save_artifacts=True
                    )
                    runs.append(res.test_metrics)
                summary_rows[f"{m_id} ({label})"] = aggregate_seed_metrics(runs)

        table_md = f"# E4 Sensor Noise Robustness\n\n{format_results_table(summary_rows)}\n"
        payload = {"experiment_id": "E4", "aggregated": summary_rows, "table_markdown": table_md}
        _write_outputs(root, "E4", payload, table_md)
        return payload

    if exp_key == "E5":
        run_seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]
        folds = all_lovo_folds(df)
        target_vids = vehicles_subset if vehicles_subset is not None else sorted(folds.keys())
        fold_runs: dict[str, dict[str, list[dict[str, float]]]] = {"A": {}, "B": {}}
        table_rows: dict[str, dict[str, Any]] = {}

        for vid in target_vids:
            f_split = folds[vid]
            prepared = prepare_datasets(df, f_split, t_seq=t_seq, stride=stride)
            for m_id in ("A", "B"):
                fold_runs[m_id][vid] = []
                for s in run_seeds:
                    cfg = TrainerConfig(
                        model_id=m_id,
                        T_seq=t_seq,
                        stride=stride,
                        batch_size=batch_size,
                        hidden_dim=hidden_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        learning_rate=learning_rate,
                        epochs=epochs,
                        warmup_epochs=warmup_epochs,
                        lambda_max=lambda_max,
                        seed=s,
                    )
                    res = Trainer(cfg, prepared, root_dir=root).fit(
                        run_id=f"E5_holdout_{vid}_model_{m_id}_seed{s}", save_artifacts=True
                    )
                    fold_runs[m_id][vid].append(res.test_metrics)
                table_rows[f"{m_id} ({vid})"] = aggregate_seed_metrics(fold_runs[m_id][vid])

        summary_a = summarize_lovo_folds(fold_runs["A"])
        summary_b = summarize_lovo_folds(fold_runs["B"])
        table_rows["A (Macro Avg)"] = summary_a["macro_avg"]
        table_rows["B (Macro Avg)"] = summary_b["macro_avg"]

        table_md = f"# E5 Leave-One-Vehicle-Out (Per-Fold & Macro)\n\n{format_results_table(table_rows)}\n"
        payload = {
            "experiment_id": "E5",
            "model_A": summary_a,
            "model_B": summary_b,
            "table_markdown": table_md,
        }
        _write_outputs(root, "E5", payload, table_md)
        return payload

    if exp_key == "E6":
        run_seeds = seeds if seeds is not None else [0, 1, 2, 3, 4]
        regime_split = build_cross_regime_splits(df)
        prepared = prepare_datasets(df, regime_split, t_seq=t_seq, stride=stride)
        summary_rows = {}
        for m_id in ("A", "B"):
            runs = []
            for s in run_seeds:
                cfg = TrainerConfig(
                    model_id=m_id,
                    T_seq=t_seq,
                    stride=stride,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    warmup_epochs=warmup_epochs,
                    lambda_max=lambda_max,
                    seed=s,
                )
                res = Trainer(cfg, prepared, root_dir=root).fit(
                    run_id=f"E6_cross_regime_model_{m_id}_seed{s}", save_artifacts=True
                )
                runs.append(res.test_metrics)
            summary_rows[f"Model {m_id}"] = aggregate_seed_metrics(runs)

        table_md = f"# E6 Cross-Regime Generalisation (Urban+Arterial -> Highway)\n\n{format_results_table(summary_rows)}\n"
        payload = {"experiment_id": "E6", "aggregated": summary_rows, "table_markdown": table_md}
        _write_outputs(root, "E6", payload, table_md)
        return payload

    if exp_key == "E7":
        run_seeds = seeds if seeds is not None else [0, 1, 2]
        grid = lambda_grid if lambda_grid is not None else [1e-3, 1e-2, 1e-1, 1e0, 1e1]
        prepared = prepare_datasets(df, split_ids, t_seq=t_seq, stride=stride)

        sweep_val_runs: dict[float, list[dict[str, float]]] = {}
        summary_rows = {}
        best_lam = grid[0]
        best_val_mae = float("inf")

        for lam in grid:
            sweep_val_runs[float(lam)] = []
            for s in run_seeds:
                cfg = TrainerConfig(
                    model_id="B",
                    T_seq=t_seq,
                    stride=stride,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    warmup_epochs=warmup_epochs,
                    lambda_max=float(lam),
                    seed=s,
                )
                res = Trainer(cfg, prepared, root_dir=root).fit(
                    run_id=f"E7_lam{lam:g}_seed{s}", save_artifacts=True
                )
                sweep_val_runs[float(lam)].append(res.val_metrics)

            agg = aggregate_seed_metrics(sweep_val_runs[float(lam)])
            summary_rows[f"B (lambda={lam:g})"] = agg
            if agg["range_mae_km"]["mean"] < best_val_mae:
                best_val_mae = agg["range_mae_km"]["mean"]
                best_lam = float(lam)

        if len(run_seeds) >= 3:
            plot_lambda_sweep_curve(sweep_val_runs, plots_dir / "E7_lambda_sweep.png")

        table_md = (
            f"# E7 Lambda Regularisation Sweep\n\n"
            f"{format_results_table(summary_rows)}\n\n"
            f"**Selected `lambda_max` (lowest val Range MAE):** `{best_lam:g}`\n"
        )
        payload = {
            "experiment_id": "E7",
            "selected_lambda_max": best_lam,
            "aggregated_val": summary_rows,
            "table_markdown": table_md,
        }
        _write_outputs(root, "E7", payload, table_md)
        return payload

    if exp_key == "E8":
        run_seeds = seeds if seeds is not None else [0, 1, 2]
        seq_grid = t_seq_grid if t_seq_grid is not None else [30, 60, 120]
        summary_rows = {}
        for seq_len in seq_grid:
            prepared = prepare_datasets(df, split_ids, t_seq=seq_len, stride=stride)
            runs = []
            for s in run_seeds:
                cfg = TrainerConfig(
                    model_id="B",
                    T_seq=seq_len,
                    stride=stride,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    dropout=dropout,
                    learning_rate=learning_rate,
                    epochs=epochs,
                    warmup_epochs=warmup_epochs,
                    lambda_max=lambda_max,
                    seed=s,
                )
                res = Trainer(cfg, prepared, root_dir=root).fit(
                    run_id=f"E8_Tseq{seq_len}_seed{s}", save_artifacts=True
                )
                runs.append(res.test_metrics)
            summary_rows[f"B (T_seq={seq_len})"] = aggregate_seed_metrics(runs)

        table_md = f"# E8 Sequence-Length Sweep\n\n{format_results_table(summary_rows)}\n"
        payload = {"experiment_id": "E8", "aggregated": summary_rows, "table_markdown": table_md}
        _write_outputs(root, "E8", payload, table_md)
        return payload

    if exp_key == "E9":
        sens = run_parameter_sensitivity_analysis(df, vehicles_yaml=vehicles_yaml)
        lines = [
            "# E9 Assumed Parameter Sensitivity (`C_rr`, `eta_dt`)",
            "",
            "| Tier | C_rr | eta_dt | Mean Range B (km) | Std Range B (km) | Mean P_batt (kW) | Mean End SOC |",
            "|---|---|---|---|---|---|---|",
        ]
        for tier, row in sens.items():
            lines.append(
                f"| {tier} | {row['c_rr']:.3f} | {row['eta_dt']:.2f} | "
                f"{row['mean_range_b_km']:.2f} | {row['std_range_b_km']:.2f} | "
                f"{row['mean_p_batt_kw']:.2f} | {row['mean_end_soc']:.4f} |"
            )
        table_md = "\n".join(lines) + "\n"
        payload = {"experiment_id": "E9", "sensitivity": sens, "table_markdown": table_md}
        _write_outputs(root, "E9", payload, table_md)
        return payload

    raise ValueError(f"Unsupported experiment_id={experiment_id!r}; expected E1..E9.")
