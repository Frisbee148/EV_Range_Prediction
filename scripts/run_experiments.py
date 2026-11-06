#!/usr/bin/env python3
"""CLI entry point to execute PRD §16 experiments E1–E9 from YAML configs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import load_trip_splits
from src.training.experiment_runner import load_experiment_config, run_experiment_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EV Range PINN Experiments (E1–E9).")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "experiments" / "e1_in_distribution.yaml"),
        help="Path to experiment YAML in configs/experiments/",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(ROOT / "data" / "processed" / "sumo_most_1hz.csv.gz"),
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=str(ROOT / "data" / "splits" / "trip_splits.json"),
    )
    parser.add_argument(
        "--vehicles",
        type=str,
        default=str(ROOT / "configs" / "vehicles.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    exp_cfg = load_experiment_config(args.config)
    exp_id = str(exp_cfg["experiment_id"])
    df = pd.read_csv(args.dataset)
    splits = load_trip_splits(args.splits)

    epochs = int(args.epochs) if args.epochs is not None else int(exp_cfg.get("epochs", 100))
    warmup = (
        int(args.warmup_epochs)
        if args.warmup_epochs is not None
        else int(exp_cfg.get("warmup_epochs", 10))
    )
    seeds = args.seeds if args.seeds is not None else exp_cfg.get("seeds", [0, 1, 2, 3, 4])

    out = run_experiment_suite(
        experiment_id=exp_id,
        df=df,
        split_ids=splits,
        root_dir=ROOT,
        vehicles_yaml=args.vehicles,
        seeds=[int(s) for s in seeds],
        epochs=epochs,
        warmup_epochs=warmup,
        hidden_dim=int(exp_cfg.get("hidden_dim", 128)),
        num_layers=int(exp_cfg.get("num_layers", 2)),
        dropout=float(exp_cfg.get("dropout", 0.1)),
        batch_size=int(exp_cfg.get("batch_size", 256)),
        learning_rate=float(exp_cfg.get("learning_rate", 1e-3)),
        t_seq=int(exp_cfg.get("T_seq", 60)),
        stride=int(exp_cfg.get("stride", 5)),
        lambda_max=float(exp_cfg.get("lambda_max", 1.0)),
        ablation_fractions=exp_cfg.get("ablation_fractions"),
        soc_k_intervals=exp_cfg.get("soc_k_intervals"),
        lambda_grid=exp_cfg.get("lambda_grid"),
        t_seq_grid=exp_cfg.get("t_seq_grid"),
    )
    print(out.get("table_markdown", f"Completed {exp_id}"))


if __name__ == "__main__":
    main()
