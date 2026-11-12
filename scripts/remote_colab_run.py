"""Remote Colab runner for improved PINN experiments (E2 proportional allocation + E5 LOVO + enhanced PINN)."""

import os
import subprocess
import sys
import time
import json
from pathlib import Path

REPO_URL = "https://github.com/Frisbee148/EV_Range_Prediction.git"
REPO_DIR = Path("/content/EV_Range_Prediction")

if not REPO_DIR.exists():
    print(f"Cloning {REPO_URL} into {REPO_DIR} ...")
    subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
else:
    print(f"Updating {REPO_DIR} ...")
    subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only"], check=True)

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import torch
import pandas as pd
from src.data.dataset import prepare_datasets, load_trip_splits
from src.training.trainer import Trainer, TrainerConfig
from src.evaluation.metrics import aggregate_seed_metrics, format_results_table
from src.evaluation.plots import plot_data_ablation_curve
from src.evaluation.cross_vehicle import all_lovo_folds, summarize_lovo_folds

print("PyTorch CUDA available:", torch.cuda.is_available(), "Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

DATASET_PATH = REPO_DIR / "data" / "processed" / "sumo_most_1hz.csv.gz"
SPLITS_PATH = REPO_DIR / "data" / "splits" / "trip_splits.json"

df = pd.read_csv(DATASET_PATH)
splits = load_trip_splits(SPLITS_PATH)
print(f"Loaded dataset: {df.shape[0]:,} rows across {df['trip_id'].nunique()} trips")

SEEDS = [0, 1, 2, 3, 4]

# -------------------------------------------------------------------------
# Part 1: Re-run E2 with proportional allocation across all fractions & seeds
# -------------------------------------------------------------------------
print("\n" + "="*60)
print("RUNNING E2 (DATA ABLATION) ON COLAB GPU")
print("="*60)
fractions = [0.05, 0.10, 0.25, 0.50, 1.00]
ablation_runs = {}
summary_rows = {}

t0 = time.perf_counter()
for frac in fractions:
    ablation_runs[frac] = {"A": [], "B": []}
    for s in SEEDS:
        prepared = prepare_datasets(
            df, splits, t_seq=60, stride=5,
            ablation_fraction=frac, ablation_seed=s,
        )
        for m_id in ["A", "B"]:
            cfg = TrainerConfig(
                model_id=m_id, T_seq=60, stride=5, batch_size=64,
                hidden_dim=128, num_layers=2, dropout=0.1, learning_rate=1e-3,
                epochs=100, warmup_epochs=10, lambda_max=0.01, patience=15,
                seed=s, ablation_fraction=frac,
            )
            run_id = f"E2_frac{int(round(frac * 100))}_model_{m_id}_seed{s}"
            res = Trainer(cfg, prepared, root_dir=REPO_DIR).fit(run_id=run_id, save_artifacts=True)
            ablation_runs[frac][m_id].append(res.test_metrics)
            print(
                f"[{run_id}] trips={len(prepared.split_ids['train']):3d} ep={res.best_epoch:2d}/{res.epochs_run:2d} "
                f"Range MAE={res.test_metrics['range_mae_km']:6.2f} km | "
                f"Power MAE={res.test_metrics['power_mae_kw']:5.2f} kW | "
                f"Res RMS={res.test_metrics['residual_rms']:.5f}"
            )
    summary_rows[f"A ({int(round(frac * 100))}%)"] = aggregate_seed_metrics(ablation_runs[frac]["A"])
    summary_rows[f"B ({int(round(frac * 100))}%)"] = aggregate_seed_metrics(ablation_runs[frac]["B"])

e2_table_md = f"# E2 Data Ablation (Proportional Trip Allocation)\n\n{format_results_table(summary_rows)}\n"
(REPO_DIR / "results" / "tables" / "E2.md").write_text(e2_table_md, encoding="utf-8")
e2_payload = {
    "experiment_id": "E2",
    "fractions": [float(f) for f in fractions],
    "aggregated": summary_rows,
    "table_markdown": e2_table_md,
}
(REPO_DIR / "results" / "metrics" / "E2.json").write_text(json.dumps(e2_payload, indent=2), encoding="utf-8")
plot_data_ablation_curve(ablation_runs, REPO_DIR / "results" / "plots" / "E2_data_ablation.png")
print(f"E2 complete in {time.perf_counter() - t0:.1f}s")

# -------------------------------------------------------------------------
# Part 2: Re-run E5 (Leave-One-Vehicle-Out) with min_ckpt_epoch >= 1
# -------------------------------------------------------------------------
print("\n" + "="*60)
print("RUNNING E5 (LEAVE-ONE-VEHICLE-OUT) ON COLAB GPU")
print("="*60)
folds = all_lovo_folds(df)
fold_runs = {"A": {}, "B": {}}
e5_table_rows = {}

t1 = time.perf_counter()
for vid, f_split in sorted(folds.items()):
    prepared = prepare_datasets(df, f_split, t_seq=60, stride=5)
    for m_id in ("A", "B"):
        fold_runs[m_id][vid] = []
        for s in SEEDS:
            cfg = TrainerConfig(
                model_id=m_id, T_seq=60, stride=5, batch_size=64,
                hidden_dim=128, num_layers=2, dropout=0.1, learning_rate=1e-3,
                epochs=100, warmup_epochs=10, lambda_max=0.01, patience=15,
                seed=s,
            )
            res = Trainer(cfg, prepared, root_dir=REPO_DIR).fit(
                run_id=f"E5_holdout_{vid}_model_{m_id}_seed{s}", save_artifacts=True
            )
            fold_runs[m_id][vid].append(res.test_metrics)
            print(
                f"[{vid} {m_id} s={s}] ep={res.best_epoch:2d}/{res.epochs_run:2d} "
                f"Range MAE={res.test_metrics['range_mae_km']:6.2f} km | "
                f"Power MAE={res.test_metrics['power_mae_kw']:5.2f} kW | "
                f"Res RMS={res.test_metrics['residual_rms']:.5f}"
            )
        e5_table_rows[f"{m_id} ({vid})"] = aggregate_seed_metrics(fold_runs[m_id][vid])

summary_a = summarize_lovo_folds(fold_runs["A"])
summary_b = summarize_lovo_folds(fold_runs["B"])
e5_table_rows["A (Macro Avg)"] = summary_a["macro_avg"]
e5_table_rows["B (Macro Avg)"] = summary_b["macro_avg"]

e5_table_md = f"# E5 Leave-One-Vehicle-Out (Updated Checkpointing)\n\n{format_results_table(e5_table_rows)}\n"
(REPO_DIR / "results" / "tables" / "E5.md").write_text(e5_table_md, encoding="utf-8")
e5_payload = {
    "experiment_id": "E5",
    "model_A": summary_a,
    "model_B": summary_b,
    "table_markdown": e5_table_md,
}
(REPO_DIR / "results" / "metrics" / "E5.json").write_text(json.dumps(e5_payload, indent=2), encoding="utf-8")
print(f"E5 complete in {time.perf_counter() - t1:.1f}s")

# -------------------------------------------------------------------------
# Part 3: Enhanced PINN Demonstration (Model B+ with L4 & full warmup)
# -------------------------------------------------------------------------
print("\n" + "="*60)
print("RUNNING ENHANCED PINN (MODEL B+ WITH L4 RANGE CONSISTENCY)")
print("="*60)
t2 = time.perf_counter()
enhanced_runs = {"Model_A": [], "Model_B_default": [], "Model_B_enhanced": []}
for s in SEEDS:
    prep_enh = prepare_datasets(
        df, splits, t_seq=60, stride=5,
        ablation_fraction=0.25, noise_enabled=True, noise_seed=42, ablation_seed=s,
    )
    # 1. Model A
    cfg_a = TrainerConfig(
        model_id="A", T_seq=60, stride=5, batch_size=64,
        epochs=50, warmup_epochs=10, lambda_max=0.0, patience=15, seed=s,
    )
    res_a = Trainer(cfg_a, prep_enh, root_dir=REPO_DIR).fit(save_artifacts=False)
    enhanced_runs["Model_A"].append(res_a.test_metrics)

    # 2. Model B Default
    cfg_b = TrainerConfig(
        model_id="B", T_seq=60, stride=5, batch_size=64,
        epochs=50, warmup_epochs=10, lambda_max=0.01, patience=15,
        use_range_consistency=False, min_ckpt_epoch=1, seed=s,
    )
    res_b = Trainer(cfg_b, prep_enh, root_dir=REPO_DIR).fit(save_artifacts=False)
    enhanced_runs["Model_B_default"].append(res_b.test_metrics)

    # 3. Model B Enhanced (L4 + full warmup)
    cfg_b_plus = TrainerConfig(
        model_id="B", T_seq=60, stride=5, batch_size=64,
        epochs=50, warmup_epochs=10, lambda_max=0.05, patience=15,
        use_range_consistency=True, c_r=0.2, min_ckpt_epoch=10, seed=s,
    )
    res_b_plus = Trainer(cfg_b_plus, prep_enh, root_dir=REPO_DIR).fit(save_artifacts=False)
    enhanced_runs["Model_B_enhanced"].append(res_b_plus.test_metrics)
    print(
        f"[Seed {s}] Range MAE: A={res_a.test_metrics['range_mae_km']:.2f} km | "
        f"B_default={res_b.test_metrics['range_mae_km']:.2f} km | "
        f"B_enhanced={res_b_plus.test_metrics['range_mae_km']:.2f} km"
    )

enh_summary = {k: aggregate_seed_metrics(v) for k, v in enhanced_runs.items()}
enh_table_md = f"# Enhanced PINN Benchmark (25% Data + Sensor Noise)\n\n{format_results_table(enh_summary)}\n"
(REPO_DIR / "results" / "tables" / "enhanced_pinn_comparison.md").write_text(enh_table_md, encoding="utf-8")
(REPO_DIR / "results" / "metrics" / "enhanced_pinn.json").write_text(json.dumps(enh_summary, indent=2), encoding="utf-8")
print(f"Enhanced PINN complete in {time.perf_counter() - t2:.1f}s")

# Archive results
subprocess.run(["tar", "-czf", "/content/updated_results.tar.gz", "-C", str(REPO_DIR), "results"], check=True)
print("\nALL RUNS COMPLETE! Results archived to /content/updated_results.tar.gz")
