"""TDD tests for Phase 1–5 audit gaps (PRD §9.4, §19.3, §22.6) and Phase 6 Experiment Suite (PRD §16)."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import load_trip_splits, prepare_datasets
from src.data.degradation import subsample_training_trips
from src.evaluation.evaluate import evaluate_baselines
from src.physics.verify_against_sumo import (
    compare_to_sumo_battery,
    parse_sumo_battery_xml,
    sumo_battery_device_power,
)
from src.training.experiment_runner import (
    build_cross_regime_splits,
    load_experiment_config,
    run_experiment_suite,
    run_parameter_sensitivity_analysis,
)
from src.vehicles import load_all_vehicles, load_physics_constants, load_vehicle

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "sumo_most_1hz.csv.gz"
SPLITS_PATH = ROOT / "data" / "splits" / "trip_splits.json"
VEHICLES_PATH = ROOT / "configs" / "vehicles.yaml"
EXP_CONFIG_DIR = ROOT / "configs" / "experiments"


def test_prd_19_3_no_hardcoded_vehicle_params_in_physics():
    """PRD §19.3: grep/AST check that src/physics/ contains no hard-coded vehicle parameters."""
    forbidden_names = {
        "battery_capacity_kwh",
        "vehicle_mass_kg",
        "drag_coeff",
        "frontal_area_m2",
        "rolling_resistance",
        "drivetrain_efficiency",
        "aux_power_kw",
    }
    forbidden_literals = {1750.0, 42.2, 0.28, 2.2, 0.92, 0.8}

    physics_dir = ROOT / "src" / "physics"
    for py_file in physics_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assert target.id not in forbidden_names, (
                            f"Forbidden hard-coded variable {target.id!r} in {py_file.name}"
                        )
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                assert float(node.value) not in forbidden_literals, (
                    f"Forbidden hard-coded numeric literal {node.value} in {py_file.name}"
                )


def test_prd_22_full_dataset_gates_elevation_and_regen_clamp():
    """PRD §22.3, §22.6: verify full MoST dataset satisfies std(z) > 1.0 and clamped regen."""
    df = pd.read_csv(DATASET_PATH)
    vehicles = load_all_vehicles(VEHICLES_PATH)

    assert float(df["elevation_m"].std()) > 1.0
    for vid, group in df.groupby("vehicle_id"):
        v_cfg = vehicles[str(vid)]
        min_p = float(group["p_batt_w"].min())
        assert min_p >= -(v_cfg.p_regen_max_w + 1e-3), (
            f"Unclamped regen for {vid}: min(p_batt_w)={min_p} < -{v_cfg.p_regen_max_w}"
        )


def test_prd_9_4_sumo_battery_device_crosscheck(tmp_path: Path):
    """PRD §9.4: verify physics engine agrees with SUMO battery device within 2%."""
    vehicle = load_vehicle(VEHICLES_PATH, "bmw_i3_60ah")
    constants = load_physics_constants(VEHICLES_PATH)

    df = pd.read_csv(DATASET_PATH)
    trip = (
        df[df["vehicle_id"] == "bmw_i3_60ah"]
        .groupby("trip_id")
        .first()
        .index[0]
    )
    trip_df = df[df["trip_id"] == trip].sort_values("timestamp")

    report = compare_to_sumo_battery(
        v_mps=trip_df["speed_mps"].to_numpy(),
        a_mps2=trip_df["accel_mps2"].to_numpy(),
        theta_rad=trip_df["grade_rad"].to_numpy(),
        t_amb_c=trip_df["ambient_temp_c"].to_numpy(),
        soc_0=float(trip_df["soc"].iloc[0]),
        dt_s=1.0,
        vehicle=vehicle,
        constants=constants,
        tolerance_rel=0.02,
    )
    assert report["passes_2pct_gate"] is True
    assert report["relative_energy_diff"] <= 0.02

    # Also verify XML parser on a synthetic SUMO --battery-output XML
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<battery-export>
    <timestep time="0.00">
        <vehicle id="ev_0" energyConsumed="0.00" actualBatteryCapacity="15000.0" maximumBatteryCapacity="18800.0" speed="10.0" acceleration="1.0" slope="0.0"/>
    </timestep>
    <timestep time="1.00">
        <vehicle id="ev_0" energyConsumed="5.20" actualBatteryCapacity="14994.8" maximumBatteryCapacity="18800.0" speed="11.0" acceleration="1.0" slope="0.0"/>
    </timestep>
</battery-export>
"""
    xml_path = tmp_path / "battery_out.xml"
    xml_path.write_text(xml_content, encoding="utf-8")
    parsed = parse_sumo_battery_xml(xml_path)
    assert len(parsed) == 2
    assert np.isclose(float(parsed["energy_consumed_wh"].iloc[1]), 5.20)


def test_all_experiment_yaml_configs_exist_and_load():
    """PRD §16, §18: verify one YAML per experiment E1–E9 exists in configs/experiments/."""
    expected_files = [
        "e1_in_distribution.yaml",
        "e2_data_ablation.yaml",
        "e3_sparse_soc.yaml",
        "e4_noise_robustness.yaml",
        "e5_leave_one_vehicle_out.yaml",
        "e6_cross_regime.yaml",
        "e7_lambda_sweep.yaml",
        "e8_seq_length_sweep.yaml",
        "e9_param_sensitivity.yaml",
    ]
    for fname in expected_files:
        path = EXP_CONFIG_DIR / fname
        assert path.exists(), f"Missing experiment config: {path}"
        cfg = load_experiment_config(path)
        assert "experiment_id" in cfg
        assert "description" in cfg


def test_cross_regime_splits_and_parameter_sensitivity():
    """PRD §16.6 (E6) and §16.9 (E9)."""
    df = pd.read_csv(DATASET_PATH)
    regime_splits = build_cross_regime_splits(
        df, train_scenarios=("urban", "arterial"), test_scenarios=("highway",), val_frac=0.15, seed=42
    )
    train_scen = set(df[df["trip_id"].isin(set(regime_splits["train"]))]["scenario"].unique())
    val_scen = set(df[df["trip_id"].isin(set(regime_splits["val"]))]["scenario"].unique())
    test_scen = set(df[df["trip_id"].isin(set(regime_splits["test"]))]["scenario"].unique())
    assert train_scen <= {"urban", "arterial"}
    assert val_scen <= {"urban", "arterial"}
    assert test_scen == {"highway"}

    sub_df = df[df["trip_id"].isin(sorted(df["trip_id"].unique())[:8])].copy()
    sens = run_parameter_sensitivity_analysis(sub_df, VEHICLES_PATH)
    assert set(sens.keys()) == {"low", "mid", "high"}
    assert sens["low"]["mean_range_b_km"] > sens["high"]["mean_range_b_km"]


def test_experiment_runner_smoke_e1_and_e7(tmp_path: Path):
    """Verify experiment orchestrator runs E1 and E7 end-to-end and emits tables, metrics, and plots."""
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

    e1_out = run_experiment_suite(
        experiment_id="E1",
        df=sub_df,
        split_ids=small_splits,
        root_dir=tmp_path,
        vehicles_yaml=VEHICLES_PATH,
        seeds=[0, 1, 2],
        epochs=2,
        warmup_epochs=1,
        hidden_dim=32,
        batch_size=32,
        stride=15,
    )
    assert "table_markdown" in e1_out
    assert "paired_A_minus_B" in e1_out
    assert (tmp_path / "results" / "metrics" / "E1.json").exists()
    assert (tmp_path / "results" / "tables" / "E1.md").exists()
    assert (tmp_path / "results" / "plots" / "E1_comparison_bars.png").exists()

    e7_out = run_experiment_suite(
        experiment_id="E7",
        df=sub_df,
        split_ids=small_splits,
        root_dir=tmp_path,
        vehicles_yaml=VEHICLES_PATH,
        seeds=[0, 1, 2],
        epochs=2,
        warmup_epochs=1,
        hidden_dim=32,
        batch_size=32,
        stride=15,
        lambda_grid=[1e-2, 1e0],
    )
    assert "selected_lambda_max" in e7_out
    assert (tmp_path / "results" / "metrics" / "E7.json").exists()
    assert (tmp_path / "results" / "plots" / "E7_lambda_sweep.png").exists()


def test_ablation_fractions_produce_distinct_trip_counts_and_base0_range_is_accurate():
    """Regression test for Issues 2 and 3:
    - 5% and 10% ablation fractions must produce distinct trip counts (8 vs 17 of 168).
    - Base-0 evaluated with test_trip_frames must match Definition B range (< 1 km MAE),
      and window-only Base-0 must stay bounded (<= 1000 km).
    """
    df = pd.read_csv(DATASET_PATH)
    splits = load_trip_splits(SPLITS_PATH)

    trips_5 = subsample_training_trips(df, splits["train"], fraction=0.05, seed=0)
    trips_10 = subsample_training_trips(df, splits["train"], fraction=0.10, seed=0)
    trips_25 = subsample_training_trips(df, splits["train"], fraction=0.25, seed=0)
    assert len(trips_5) == round(len(splits["train"]) * 0.05)
    assert len(trips_10) == round(len(splits["train"]) * 0.10)
    assert len(trips_5) < len(trips_10) < len(trips_25)
    assert trips_5 != trips_10

    small_splits = {
        "train": splits["train"][:6],
        "val": splits["val"][:3],
        "test": splits["test"][:3],
    }
    sub_df = df[
        df["trip_id"].isin(
            set(small_splits["train"]) | set(small_splits["val"]) | set(small_splits["test"])
        )
    ].copy()
    prepared = prepare_datasets(sub_df, small_splits, t_seq=60, stride=15)
    test_df = sub_df[sub_df["trip_id"].isin(set(small_splits["test"]))].copy()

    eval_full = evaluate_baselines(
        prepared.train_dataset.windows,
        prepared.test_dataset.windows,
        vehicles_yaml=VEHICLES_PATH,
        test_trip_frames=test_df,
    )
    assert eval_full["Base-0"]["metrics"]["range_mae_km"] < 0.1

    eval_win = evaluate_baselines(
        prepared.train_dataset.windows,
        prepared.test_dataset.windows,
        vehicles_yaml=VEHICLES_PATH,
    )
    assert eval_win["Base-0"]["metrics"]["range_mae_km"] < 1000.0

