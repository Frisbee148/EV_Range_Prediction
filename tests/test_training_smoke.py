"""CPU smoke-training test for Model A and Model B, windowing, and scaler checks (PRD §12, §14, §15)."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.dataset import load_trip_splits, prepare_datasets
from src.models.pinn import build_model, verify_matched_architecture
from src.training.trainer import Trainer, TrainerConfig

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "processed" / "sumo_most_1hz.csv.gz"
SPLITS_PATH = ROOT / "data" / "splits" / "trip_splits.json"


def test_matched_architecture_between_model_a_and_b():
    model_a = build_model("A", hidden_dim=64, num_layers=2, dropout=0.1)
    model_b = build_model("B", hidden_dim=64, num_layers=2, dropout=0.1)
    verify_matched_architecture(model_a, model_b)


def test_cpu_smoke_training_model_a_and_model_b(tmp_path: Path):
    df = pd.read_csv(DATASET_PATH)
    full_splits = load_trip_splits(SPLITS_PATH)

    # Select a compact subset of trips for fast CPU smoke verification (PRD §21 step 1)
    small_splits = {
        "train": full_splits["train"][:8],
        "val": full_splits["val"][:4],
        "test": full_splits["test"][:4],
    }
    subset_ids = set(small_splits["train"]) | set(small_splits["val"]) | set(small_splits["test"])
    sub_df = df[df["trip_id"].isin(subset_ids)].copy()

    prepared = prepare_datasets(
        sub_df,
        small_splits,
        t_seq=60,
        stride=10,
        soc_k=10,
    )

    # Verify PRD §22.7: scaler fitted sample count equals training split row count
    expected_train_rows = int((sub_df["trip_id"].isin(set(small_splits["train"]))).sum())
    prepared.scalers.assert_fitted_on_train_only(expected_train_rows)

    # Smoke-train Model A (Data-only LSTM)
    cfg_a = TrainerConfig(
        model_id="A",
        T_seq=60,
        stride=10,
        batch_size=64,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        learning_rate=3e-3,
        epochs=6,
        warmup_epochs=2,
        lambda_max=1.0,
        seed=0,
        device="cpu",
    )
    trainer_a = Trainer(cfg_a, prepared, root_dir=tmp_path)
    res_a = trainer_a.fit(run_id="smoke_model_A", save_artifacts=True)

    assert res_a.epochs_run == 6
    assert res_a.history[-1]["train_loss_data"] < res_a.history[0]["train_loss_data"]
    assert all(row["train_lambda_t"] == 0.0 for row in res_a.history)
    assert res_a.checkpoint_path is not None and res_a.checkpoint_path.exists()
    assert res_a.scaler_path is not None and res_a.scaler_path.exists()
    assert res_a.manifest_path is not None and res_a.manifest_path.exists()

    # Smoke-train Model B (Physics-informed LSTM) under matched budget
    cfg_b = TrainerConfig(
        model_id="B",
        T_seq=60,
        stride=10,
        batch_size=64,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        learning_rate=3e-3,
        epochs=6,
        warmup_epochs=2,
        lambda_max=1.0,
        seed=0,
        device="cpu",
    )
    trainer_b = Trainer(cfg_b, prepared, root_dir=tmp_path)
    verify_matched_architecture(trainer_a.model, trainer_b.model)
    res_b = trainer_b.fit(run_id="smoke_model_B", save_artifacts=True)

    assert res_b.epochs_run == 6
    assert res_b.history[-1]["train_loss_data"] < res_b.history[0]["train_loss_data"]
    assert res_b.history[0]["train_lambda_t"] == 0.0
    assert res_b.history[-1]["train_lambda_t"] == 1.0
    assert np.isfinite(res_b.test_metrics["range_mae_km"])
    assert np.isfinite(res_b.test_metrics["residual_rms"])
    assert res_b.test_metrics["residual_rms"] < res_a.test_metrics["residual_rms"]

    with res_b.manifest_path.open("r", encoding="utf-8") as fh:
        manifest_b = yaml.safe_load(fh)
    assert manifest_b["hyperparameters"]["model_id"] == "B"
    assert "L2_consist" in manifest_b["physics_loss"]["components_enabled"]
